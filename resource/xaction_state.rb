#!/usr/bin/ruby

require '../lib/util.rb'      
require 'Date'

xaction_seq_file = ARGV[0]

File.open(xaction_seq_file, "r").each_line do |line|
	items =  line.split(",")
	custID = items[0]
	row = custID	
	if (items.size >= 5)
		seq = []
		i = 4
		while i < items.size do
			amt = items[i].to_i
			prAmt = items[i-2].to_i
			
			date = Date.parse items[i-1]
			prDate = Date.parse items[i-3]
			
			daysDiff = date - prDate
			amtDiff = amt - prAmt

			if (daysDiff < 30)
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
			
			i += 2
		end
	
		puts "#{custID},#{seq.join(',')}"
		
	end
end
