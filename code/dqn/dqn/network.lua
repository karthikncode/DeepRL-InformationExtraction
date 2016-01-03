
require "initenv"

function create_network(args)

    local net = nn.Sequential()

    net:add(nn.Linear(args.state_dim, args.n_hid))
    net:add(args.nl())
    net:add(nn.Linear(args.n_hid, args.n_hid))
    net:add(args.nl())

    parallel_net = nn.ConcatTable()    
    parallel_net:add(nn.Linear(args.n_hid, args.n_actions))
    parallel_net:add(nn.Linear(args.n_hid, args.n_objects))

    net:add(parallel_net)

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
    end
    return net
end
