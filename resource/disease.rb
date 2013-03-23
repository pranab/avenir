#!/usr/bin/ruby

require '../lib/util.rb'      

userCount = ARGV[0].to_i


idGen = IdGenerator.new
race_dist = CategoricalField.new("EUA",10,"AFA",3,"LAA", 1, "ASA", 1)
diet_dist = CategoricalField.new("LF",2,"REG",8,"HF", 4)
fam_hist_dist = CategoricalField.new("NFH",5,"FH",1)
domestic_life_dist = CategoricalField.new("S",2,"DP",4)


1.upto userCount do 
	id = userID = idGen.generate(12)
	age = 20 + rand(60)
	race = race_dist.value
	weight = 120 + rand(120)
	diet = diet_dist.value
	fam_hist = fam_hist_dist.value
	domestic_life = domestic_life_dist.value
	
	pr = 15.0
	if (age < 40)
		pr *= 1.0
	elsif (age < 50)
		pr *= 1.05
	elsif (age < 60)
		pr *= 1.15
	elsif (age < 70)
		pr *= 1.4
	else
		pr *= 1.5
	end
	
	case race
		when "AFA"
		pr *= 1.2
		
		when "ASA"
		pr *= 0.9

		when "LAA"
		pr *= 0.95
	end
	
	case diet
		when "LF"
		pr *= 1.0

		when "HF"
		pr *= 1.15
	end

	case fam_hist
		when "FH"
		pr *= 1.2
	end
	
	case domestic_life
		when "S"
		pr *= 1.2
	end
	
	pr = pr > 99 ? 99 : pr
	if (rand(100) < pr)
		status = "Yes"
	else 
		status = "No"
	end
	
	puts "#{id},#{age},#{race},#{weight},#{diet},#{fam_hist},#{domestic_life},#{status}"

end
