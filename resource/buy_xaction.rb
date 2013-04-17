#!/usr/bin/ruby

require '../lib/util.rb'      
require 'Date'

custCount = ARGV[0].to_i
daysCount = ARGV[1].to_i
visitorPercent = ARGV[2].to_f

custIDs = []
xactionHist = {}

# transition probability matrix

idGen = IdGenerator.new
1.upto custCount do
	custIDs << idGen.generate(10)
end

xid = Time.now().to_i

date = Date.parse "2013-01-01"
1.upto daysCount do 
	numXaction = visitorPercent * custCount
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
			if (numDays < 30) 
				amount = lastAmt < 40 ? 50 + rand(20) - 10  : 30 + rand(10) - 5
			elsif (numDays < 60)
				amount = lastAmt < 80 ? 100 + rand(40) - 20 : 60 + rand(20) - 10
			else 
				amount = lastAmt < 150 ? 180 + rand(60) - 30 :  120 + rand(40) - 20
			end						
		else
			hist = []
			xactionHist[custID] = hist	
			amount = 40 + rand(180)
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
