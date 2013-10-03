#!/usr/bin/ruby

require '../lib/util.rb'      

custCount = ARGV[0].to_i
events = ['SL','SS','SM','ML','MS','MM','LL','LS','LM']


idGen = IdGenerator.new
1.upto custCount do
	custID = idGen.generate(10)
	num_events = 5 + rand(20)
	cust_events = []
	1.upto num_events do
		indx = rand(events.size)
		ev = events[indx]
		cust_events << ev
		if (rand(10) < 3)
			r = 1 + rand(3)
			1.upto r do
				indx = (indx / 3) * 3 + rand(2)
				ev = events[indx]
				cust_events << ev
			end
		end
	end
	
	puts "#{custID},#{cust_events.join(',')}"

end
