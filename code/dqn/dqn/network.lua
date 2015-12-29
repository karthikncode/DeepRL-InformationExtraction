
require "initenv"

function create_network(args)

    local net = nn.Sequential()

    net:add(nn.Linear(args.state_dim, args.n_hid))
    net:add(args.nl())
    net:add(nn.Linear(args.n_hid, args.n_hid))
    net:add(args.nl())
    -- net:add(nn.Linear(args.n_hid, args.n_hid))
    -- net:add(args.nl())
    -- net:add(nn.Linear(args.n_hid, args.n_hid))
    -- net:add(args.nl())
    net:add(nn.Linear(args.n_hid, args.n_actions))

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
    end
    return net
end
