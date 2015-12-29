--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'network'

return function(args)
    args.n_hid          = 20
    args.nl             = nn.Rectifier

    return create_network(args)
end

