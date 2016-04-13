
local Rectifier, parent = torch.class('nn.Rectifier', 'nn.Module')

-- This module accepts minibatches
function Rectifier:updateOutput(input)
    return self.output:resizeAs(input):copy(input):abs():add(input):div(2)
end

function Rectifier:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(self.output)
    return self.gradInput:sign(self.output):cmul(gradOutput)
end