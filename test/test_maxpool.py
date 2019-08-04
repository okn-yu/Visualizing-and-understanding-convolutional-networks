import torch
import torch.nn as nn

pool = nn.MaxPool2d((2, 2), stride=1, padding=0, return_indices=True)
input = torch.arange(0, 9, dtype=torch.float32).view(1, 1, 3, 3)

print(input)
print(input.shape) # torch.Size([1, 1, 4, 4])

output, indices = pool(input)
print(output)       # 4*4の行列の最大値
print(output.shape) # torch.Size([1, 1, 2, 2])
print(indices)      # 4*4の行列を1次元にしたインデックスを考える

unpool = nn.MaxUnpool2d((2, 2), stride=1, padding=0)

result1 = unpool(output, indices)
#result2 = unpool(output, indices, output_size=torch.Size([1, 1, 5, 5]))

print(result1) # 最大値は同じ値、それ以外は全て0
#print(result1.shape)
#print(result2)
#print(result2.shape)