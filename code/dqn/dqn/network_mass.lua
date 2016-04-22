
require 'network'

return function(args)
    args.n_hid          = 20
    -- args.nl             = nn.Sigmoid
    args.nl             = nn.Rectifier

    return create_network(args)
end

