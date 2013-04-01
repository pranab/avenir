#!/usr/bin/ruby

require '../lib/util.rb'      

custCount = ARGV[0].to_i
daysCount = ARGV[1].to_i
custIDs = []
xactionHist = {}

# transition probability matrix

IdGenerator idGen = IdGenerator.new
1.upto custCount do
	custIDs << idGen.generate(10)
end

date = Date.parse "2012-01-01"
1.upto daysCount do 
	numXaction = .02 * custCount
	factor = 85 + rand(30)
	numXaction = (numXaction * factor) / 100
	
	1.upto numXaction do
		custID = custIDs[rand(custIDs.size)]
		if (xactionHist.includes? custID)
			hist = xactionHist[custID]
			lastXaction = hist[-1]
			lastDate = lastXaction[0]
			numDays = date - lastDate
			if (numDays < 15) 
				amount = 10 + rand(40)
			elsif (numDays < 60)
				amount = 40 + rand(80)
			else 
				amount = 60 + rand(140)
			end						
		else
			hist = []
			xactionHist[custID] = hist	
			amount = 20 + rand(180)
		end
		xaction = []
		xaction << date
		xaction << amount
		hist << xaction
		puts "#{custID},#{date},#{amount}"
	end

	date = date.next
end
