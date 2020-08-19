# Torchmetaの使い方
このドキュメントは公式ドキュメントの追加資料として、MAMLおよびMLDGをtorchmeta libraryを使って実装するための最低限の説明書である。  

## **関連文書**
* 公式ドキュメント  
    https://github.com/tristandeleu/pytorch-meta/tree/master/torchmeta  

    https://pypi.org/project/torchmeta/#data  

* arxiv  
    https://arxiv.org/abs/1909.06576  

* medium  
    https://medium.com/pytorch/torchmeta-a-meta-learning-library-for-pytorch-f76c2b07ca6d


## **重要機能**

### **1. Meta conv, Meta linear**
torch.convやtorch.linearとの違いは入力にパラメタ（W,b）を受け付けること。  
入力した重みとバイアスでConvを計算し結果を返す。

```python
 class MetaConv2d(nn.Conv2d, MetaModule):
    __doc__ = nn.Conv2d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            params['weight'], bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, params['weight'], bias, self.stride,
                        self.padding, self.dilation, self.groups)
```

### **2. torchmeta.utils.gradient_based()**  

Auto gradしてSGDでパラメタをアップデートする機能  
MAMLのinner loop，MLDGのMeta trainの更新部分に当たる。
```python
import torch

from collections import OrderedDict
from torchmeta.modules import MetaModule


def gradient_update_parameters(model,
                               loss,
                               params=None,
                               step_size=0.5,
                               first_order=False):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.
    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.
    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.
    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.
    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.
    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the inner-loss.
    """
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

    return updated_params
```

## **torchmetaを用いた MAMLの実装**  

公式  
https://github.com/tristandeleu/pytorch-maml  

modelをmeta conv, meta linearを使って構築（nn.Sequentialを使うときは別途 MetaSequentialを使う必要があるので注意）  

inner loopでの実装方法例↓
modelにinput, paramを入力し通常通りLossを計算し、modelとLoss, paramを入力にgradient_update_parameters()でparamを更新する。
inner loopの出力には更新後のparamとinner losssとaccを辞書形式で返す。

```python

    def adapt(self, inputs, targets, is_classification_task=None,
              num_adaptation_steps=1, step_size=0.1, first_order=False):
        if is_classification_task is None:
            is_classification_task = (not targets.dtype.is_floating_point)
        params = None

        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            logits = self.model(inputs, params=params)
            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()

            if (step == 0) and is_classification_task:
                results['accuracy_before'] = compute_accuracy(logits, targets)

            self.model.zero_grad()
            params = gradient_update_parameters(self.model, inner_loss,
                step_size=step_size, params=params,
                first_order=(not self.model.training) or first_order)

        return params, results
```
  

outer loopの実装例  
inner loopで更新したparamsとtest inputを入力してouter lossを算出しΣを計算する。この時タスク数でLossが増えてしまうのでタスク数で除算する。

```python
 def get_outer_loss(self, batch):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        is_classification_task = (not test_targets.dtype.is_floating_point)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }
        if is_classification_task:
            results.update({
                'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
                'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
            })

        mean_outer_loss = torch.tensor(0., device=self.device)
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            params, adaptation_results = self.adapt(train_inputs, train_targets,
                is_classification_task=is_classification_task,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size, first_order=self.first_order)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            if is_classification_task:
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                outer_loss = self.loss_function(test_logits, test_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            if is_classification_task:
                results['accuracies_after'][task_id] = compute_accuracy(
                    test_logits, test_targets)

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results


```

最終的な更新部分の実装例  
outer loopでのLossで更新するだけ。（特に書くことなかった）

```python
    def train_iter(self, dataloader, max_batches=500):
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                self.optimizer.zero_grad()

                batch = tensors_to_device(batch, device=self.device)
                outer_loss, results = self.get_outer_loss(batch)
                yield results

                outer_loss.backward()
                self.optimizer.step()

                num_batches += 1
```