#!/usr/bin/ruby

require '../lib/util.rb'      
require 'Date'

xaction_file = ARGV[0]
userXactions = {}

File.open(xaction_file, "r").each_line do |line|
	items =  line.split(",")
	custID = items[0]
	if (userXactions.key? custID)
		hist = userXactions[custID]
	else 
		hist = []
		userXactions[custID] = hist
	end	
	hist << items[2, items.size - 2]
end

userXactions.each do |cid, xactions|
	seq = []
	xactions.each_with_index do |xaction, index|	
		if (index > 0) 
			prevXaction = xactions[index-1]
			prDate = Date.parse prevXaction[0]
			prAmt = prevXaction[1].to_i
			date = Date.parse xaction[0]
			amt = xaction[1].to_i
			daysDiff = date - prDate
			amtDiff = amt - prAmt
			
			if (daysDiff < 15)
				dd = "S"
			elsif (daysDiff < 60)
				dd = "M"
			else
				dd = "L"
			end
			
			if (prAmt < 0.9 * amt)
				ad = "L"
			elsif (prAmt < 1.1 * amt) 
				ad = "E"
			else 
				ad = "G"
			end
			seq << (dd + ad)
		end
	end
	if (seq.size > 1)
		puts "#{cid},#{seq.join(',')}"
	end
end

