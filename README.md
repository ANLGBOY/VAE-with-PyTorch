# Variational Auto-Encoder(VAE) for MNIST
PyTorch implementation of VAE for MNIST

## Description
This code is written for practice. It's almost as same as PyTorch's official example(check the reference below).

## Results
The following results are made with the default setting. (command: python main.py)

<table align='center'>
<tr align='center'>
<td> Reconstruction </td>
<td> Sampling </td>
<td> Manifold(num of z = 2) </td>
</tr>
<tr>
<td><img src = 'results/reconstruction_100.jpg' height = '300px'>
<td><img src = 'results/sample_100.jpg' height = '300px'>
<td><img src = 'results/plot_along_z1_and_z2_axis__100.jpg' height = '300px'>
</tr>
</table>


## References
The implementation is based on:
[1]https://github.com/pytorch/examples/blob/master/vae/main.py
[2]https://github.com/hwalsuklee/tensorflow-mnist-VAE/