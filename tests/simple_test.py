import torch
import auraloss

input = torch.rand(8, 2, 44100)
target = torch.rand(8, 2, 44100)

# loss = auraloss.freq.SumAndDifferenceSTFTLoss()
# loss = auraloss.freq.PerceptuallyWeightedComplexLoss()
# loss = auraloss.freq.MultiResolutionPrcptWghtdCmplxLoss()

# print(loss(input, target))

filter = auraloss.freq.FIRFilter(filter_type="aw", plot=True, ntaps=1001, inverse=True)

test = filter(input)