#!/usr/bin/ruby

require '../lib/util.rb'      

userCount = ARGV[0].to_i

levels = ["low", "med", "high"]


#1 minute usage (L, M, H)
#2 data usage (L, M, H)
#3 cs calls (L, M, H)
#4 account years (1,2,3,...)
#5 payment history
#6 status (open, closed)

idGen = IdGenerator.new
min_dist = CategoricalField.new("low",2,"med",5,"high", 3, "overage", 2)
data_dist = CategoricalField.new("low",4,"med",6,"high", 2)
cs_dist = CategoricalField.new("low",6,"med",3,"high", 1)
payment_dist = CategoricalField.new("poor",2,"average",5,"good", 4)

1.upto userCount do 
	id = userID = idGen.generate(12)
	min_used = min_dist.value
	data_used = data_dist.value
	cs_calls = cs_dist.value
	payment = payment_dist.value
	acct_age = rand(4) + 1
	
	pr = 25.0
	case min_used
		when "low"
		pr *= 1.2
		
		when "high"
		pr *= 1.4

		when "overage"
		pr *= 1.8
	end
	
	case data_used
		when "low"
		pr *= 1.1

		when "med"
		pr *= 1.3
		
		when "high"
		pr *= 1.6
	end

	case cs_calls
		when "med"
		pr *= 1.2
		
		when "high"
		pr *= 1.6
	end
	
	case payment
		when "poor"
		pr *= 1.3
	end
	
	case acct_age
		when 3
		pr *= 1.05
		
		when 4
		pr *= 1.2
		
		when 5
		pr *= 1.3
	end
	pr = pr > 99 ? 99 : pr
	if (rand(100) < pr)
		status = "closed"
	else 
		status = "open"
	end
	
	puts "#{id},#{min_used},#{data_used},#{cs_calls},#{payment},#{acct_age},#{status}"

end
