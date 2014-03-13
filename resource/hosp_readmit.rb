#!/usr/bin/ruby

require '../lib/util.rb'      

patientCount = ARGV[0].to_i

agetDist = NumericalFieldRange.new(10..20,2,21..30,3,31..40,6,41..50,10,51..60,14,61..70,19,71..80,25,81..90,21)
wtDist = NumericalFieldRange.new(130..140,9,141..150,13,151..160,16,161..170,20,171..180,23,181..190,20,
	191..200,17,201..211,14,211..220,10,221..230,7,231..240,5,241..250,3)
htDist = NumericalFieldRange.new(50..55,9,56..60,12,61..65,16,66..70,23,71..75,14)
empDist = CategoricalField.new('employed',10,'unemployed',1,'retired',3)
famStatDist = CategoricalField.new('alone',10,'with partner',15)
dietDist = CategoricalField.new('average',10,'poor',4,'good',2)
exerciseDist = CategoricalField.new('average',10,'low',12,'high',4)
followUpDist = CategoricalField.new('average',10,'low',14,'high',3)
smokingDist = CategoricalField.new('non smoker',10,'smoker',3)
alcoholDist = CategoricalField.new('average',10,'low',16,'high',4)

idGen = IdGenerator.new
1.upto patientCount do
	readmitProb = 20 
	patID = idGen.generate(12)
	
	age = agetDist.value
	if (age > 80)
		readmitProb = readmitProb + 10
	elsif (age > 70)
		readmitProb = readmitProb + 5
	elsif (age > 60)
		readmitProb = readmitProb + 3
	end
	
	wt = wtDist.value
	ht = htDist.value
	if (wt > 200 && ht < 70)
		readmitProb = readmitProb + 5
	elsif (wt > 180 and ht < 60)
		readmitProb = readmitProb + 3
	end
	
	emp = empDist.value
	if (age > 68 and rand(10) < 8)
		emp = 'retired'
	end
	if (emp == 'unemployed')
		readmitProb = readmitProb + 6
	elsif (emp == 'retired')
		readmitProb = readmitProb + 4
	end
		
	fam = famStatDist.value
	if (fam == 'alone')
		readmitProb = readmitProb + 9
	end
		
	diet = dietDist.value
	if (emp == 'unemployed' and rand(10) < 7)
		diet = 'poor'
	end
	if (diet == 'poor')
		readmitProb = readmitProb + 4
	elsif (diet == 'average')
		readmitProb = readmitProb + 2
	end
		
	ex = exerciseDist.value
	if (ex == 'low')
		readmitProb = readmitProb + 3
	elsif (ex == 'average')
		readmitProb = readmitProb + 1
	end
		
	followUp = followUpDist.value
	if (followUp == 'low')
		readmitProb = readmitProb + 8
	elsif (followUp == 'avearge')
		readmitProb = readmitProb + 3
	end
	
	smoking = smokingDist.value
	if (smoking == 'smoker')
		readmitProb = readmitProb + 6
	end
	
	alcohol = alcoholDist.value
	if (alcohol == 'high')
		readmitProb = readmitProb + 5
	elsif (alcohol == 'average')
		readmitProb = readmitProb + 2
	end
		
	if (rand(100) <  readmitProb)
		readmit = 'Y'
	else 
		readmit = 'N'
	end
		
	puts "#{patID},#{age},#{wt},#{ht},#{emp},#{fam},#{diet},#{ex},#{followUp},#{smoking},#{alcohol},#{readmit}"
		
end


