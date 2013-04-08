#!/usr/bin/ruby

require '../lib/util.rb'      
require 'Date'

custCount = ARGV[0].to_i
daysCount = ARGV[1].to_i
custIDs = []
xactionHist = {}

# transition probability matrix

idGen = IdGenerator.new
1.upto custCount do
	custIDs << idGen.generate(10)
end

xid = Time.now().to_i

date = Date.parse "2012-01-01"
1.upto daysCount do 
	numXaction = 0.02 * custCount
	factor = 85 + rand(30)
	numXaction = (numXaction * factor) / 100
	
	1.upto numXaction do
		custID = custIDs[rand(custIDs.size)]
		if (xactionHist.key? custID)
			hist = xactionHist[custID]
			lastXaction = hist[-1]
			lastDate = lastXaction[0]
			lastAmt = lastXaction[1]
			numDays = date - lastDate
			if (numDays < 15) 
				amount = lastAmt < 30 ? 20 + rand(50) : 10 + rand(40)
			elsif (numDays < 60)
				amount = lastAmt < 60 ? 50 + rand(80) : 30 + rand(60)
			else 
				amount = lastAmt < 120 ? 80 + rand(140) :  50 + rand(120)
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
		xid = xid + 1
		puts "#{custID},#{xid},#{date},#{amount}"
	end

	date = date.next
end
