class Room
  def initialize(value)
    puts value
  end

  def area_of_a_circle(radius)
    pi = 3.14  
    area = pi * radius * radius  
  end
end

class Knowledge
  def self.pi
    3.14  
  end

  def pi
    3.141592
  end
end

#a = Room.new('HelloWorld')
#puts a.area_of_a_circle(10)

a = Knowledge.new

