# capsule_nn

PyTorch implementation of Capsules Network, an idea from  the NIPS 2017 paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829).

## The calculation process

Here we only discuss the calculation process between the primary capsule layer and dense capsule layer.

For example, we get $10$ feature vectors which these vectors' dimensions are $16$. Thus, the shape of this tensor is $(1, 10, 8)$. The front of the shape array is $1$, because we assume that there is only one data input to the network. At the same time, we assume the output includes $5$ feature vectors which these vector's dimensions are $4$. This means that the shape of the output tensor is $(1, 5, 4)$

Then we need to affine transform the data. For one feature vector, we already the vector's shape is $(8, 1)$. The first step is use a matrix which shape is $(4, 8)$ to change the dimensions of the input feature vector's shape. After implementing the multiplication, the shape of the input feature vector is $(4, 1)$. However, we have $10$ input feature vectors, thus the affine matrix's shape should be $(10, 4, 8)$. And to implement the matrix multiplication correctly, we should change the input feature matrix's shape from $(1, 10, 8)$ to $(1, 10, 8, 1)$, which means we use the Use a column vector to represent each eigenvector.

Now we have ten features, which come from the input feature matrix, to derive a category. But we have five categories, thus we need to make $5$ times affine transform. And for this reason, the affine matrix's real shape is $(5, 10, 4, 8)$. Because we use pytorch to do this matrix multiplication, thus we need to change the input feature matrix to $(1, 1, 10, 8, 1)$. Then, pytorch will broadcast the input matrix to achieve the matrix multiplication, and we will get a matrix with shape $(1, 5, 10, 4, 1)$.

In order to implement the dynamic routing algorithm, we will initialize a matrix with shape $(1, 5, 10)$, and perform softmax operation along the dimension with index $1$ to obtain an assignment matrix. Next use the dynamic routing algorithm which is introduced in the paper to update the assignment matrix.

## Warnings
If you want to use gpu to train the model, you can't use the following code
```
model = CapsLinear((10, 8), (5, 2))
model.cuda()
# or
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
```
This is a bug, because of the existing of dynamic routing between in Capsule Linear Layer. By default, the CPU will be used for calculations. But if you want to use gpu, you have to use a specified device by passing the device name to CapsLinear. For example:
```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CapsLinear((10, 8), (5, 2), device=device)
```

In later version, we may fix this bug.
