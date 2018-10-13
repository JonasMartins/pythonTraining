
# simple puts
p "This is a string"

arr = [1,2,3,4]
# puts arr


puts 'please enter your password:'
password = gets.chomp

if password == 'password'
    p true
else
    p false
end


10.times do
    $x = 10
end

# $x makes x a global variable and possible to be accessed out of loop scope
p $x 
