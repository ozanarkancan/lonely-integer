using GPUChecker, CUDArt
CUDArt.device(first_min_used_gpu())

using Knet,JLD, ArgParse

Knet.gpu(false)

@knet function rnn(x; hidden=128, out=100)
	h = lstm(x; out=hidden)
	if predict
		return wbf(h; out=out, f=:soft)
	end
end

@knet function birnn(fx, bx; hidden=128, out=100)
	fh = lstm(fx; out=hidden)
	bh = lstm(bx; out=hidden)

	if predict
		return wbf2(fh, bh; out=out, f=:soft)
	end
end

function gendata(;seqlength=5, limit=20)
	rnums = randperm(limit)
	dup = round(Int32, (seqlength - 1) / 2)
	seq = rnums[1:dup]
	append!(seq, rnums[1:dup])
	push!(seq, rnums[dup+1])

	function onehot(indx)
		rep = zeros(Float32, limit, 1)
		rep[indx, 1] = 1.0
		return rep
	end

	onehotseq = map(onehot, seq)
	y = onehotseq[end]
	shuffle!(onehotseq)
	return (onehotseq, y)
end

function task(net; N=1024, seqlength=11, limit=100)
	@assert (seqlength % 2) == 1
	nextn = 1
	loss = 0
	loss2 = 0

	for n=1:N
		seq, ygold = gendata(;seqlength=seqlength, limit=limit)

		for i=1:(length(seq)-1); sforw(net, seq[i], seq[end-(i-1)]); end
		ypred = sforw(net, seq[end], seq[1]; predict = true)
		
		sloss = softloss(ypred, ygold)
		zloss = zeroone(ypred, ygold)

		loss = (n==1 ? sloss : 0.99 * loss + 0.01 * sloss)
		loss2 = (n==1 ? zloss : 0.99 * loss2 + 0.01 * zloss)
		n==nextn && (println((n,loss, loss2)); nextn*=2)
		flush(STDOUT)
		sback(net, ygold, softloss)
		for i=1:(length(seq)-1); sback(net); end

		update!(net; gclip=10.0)
		reset!(net)
	end
end

function timing(net; N=1024, seqlength=11, limit=100)
	seq, ygold = gendata(;seqlength=seqlength, limit=limit)

	for i=1:(length(seq)-1); sforw(net, seq[i]); end
	ypred = sforw(net, seq[end]; predict = true)

	sback(net, ygold, softloss)
	for i=1:(length(seq)-1); sback(net); end

	update!(net; gclip=10.0)
	reset!(net)
end


function test(net; N=1000, seqlength=11, limit=100)
	acc = 0
	for i=1:N
		seq, ygold = gendata(;seqlength=seqlength, limit=limit)
		for i=1:(length(seq)-1); forw(net, seq[i], seq[end-(i-1)]); end
		ypred = forw(net, seq[end], seq[i]; predict = true)
		acc += (1 - zeroone(ypred, ygold))
		reset!(net)
	end

	acc = acc / N
end

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		"--seqlength"
			help = "length of sequences"
			default = 7
			arg_type=Int
		"--N"
			help = "number of instances"
			default = 2^5
			arg_type = Int
		"--lr"
			help = "learning rate"
			default = 0.001
			arg_type = Float64
		"--limit"
			help = "upper limit of numbers"
			default = 100
			arg_type = Int
		"--hidden"
			help = "hidden size"
			default = 256
			arg_type = Int
		"--range"
			help = "testing range"
			nargs='+'
			arg_type = Int
			default = [3, 21]
	end
	return parse_args(s)
end

function main()
	args = parse_commandline()
	println("*** Params ***")
	for k in keys(args)
		println("$k -> $(args[k])")
	end

	net = compile(:birnn, hidden=args["hidden"])

	setp(net, adam=true, lr=args["lr"])

	task(net; N=args["N"], seqlength=args["seqlength"], limit=args["limit"])

	save("seq$(args["seqlength"]).jld", "net", clean(net))
	
	r = args["range"]
	for i=r[1]:2:r[2]
		acc = test(net; seqlength=i)
		println("Acc on seqlength=$i : $acc")
		flush(STDOUT)
	end
end

function timing_main()
	args = parse_commandline()
	println("*** Params ***")
	for k in keys(args)
		println("$k -> $(args[k])")
	end

	net = compile(:rnn, hidden=args["hidden"], out=args["limit"])

	setp(net, adam=true, lr=args["lr"])

	for i=1:10
		@time timing(net; N=args["N"], seqlength=args["seqlength"], limit=args["limit"])
	end
end

#main()
timing_main()
