#!/usr/bin/ruby

require '../lib/util.rb'      
require 'Date'

xaction_file = ARGV[0]
model_file = ARGV[1]
userXactions = {}
model = []
states = ["SL", "SE", "SG", "ML", "ME", "MG", "LL", "LE", "LG"]

# read all xactions
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

#read model
File.open(model_file, "r").each_line do |line|
	items =  line.split(",")
	#puts "#{line}"
	row = []
	items.each do |item|
		row << item.to_i
		#puts "#{item}"
	end	
	model << row
	#puts "#{row}"
end

# marketing time
userXactions.each do |cid, xactions|
	seq = []
	last_date = Date.parse "2000-01-01"
	xactions.each_with_index do |xaction, index|	
		if (index > 0) 
			prevXaction = xactions[index-1]
			prDate = Date.parse prevXaction[0]
			prAmt = prevXaction[1].to_i
			date = Date.parse xaction[0]
			last_date = date
			amt = xaction[1].to_i
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
		end
	end
	
	if (!seq.empty?)
		last = seq[-1]
		row_index = states.index(last)
		row = model[row_index]
		max_col = row.max
		col_index = row.index(max_col) 
		next_state = states[col_index]
		
		if (next_state.start_with?("S"))
			next_date = last_date + 15
		elsif (next_state.start_with?("M"))
			next_date = last_date + 45
		else
			next_date = last_date + 90
		end
		
		#puts "#{cid}, #{last},  #{row_index}, #{max_col}, #{col_index}, #{next_state}, #{last_date}, #{next_date}"
		puts "#{cid}, #{next_date}"
	end
end




