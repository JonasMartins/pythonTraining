puts 'Name an animal'
animal = gets.chomp
puts 'Name a noun'
noun = gets.chomp
p "The quick brown #{animal} jumped over the lazy #{noun}"

p "Show this number: #{60 + 6}"

# all string manipulation for ruby:

# http://ruby-doc.org/core-2.2.0/String.html


str = 'The quick brown fox jumped over the quick dog.'
p str.sub 'quick','slow'
# "The slow brown fox jumped over the quick dog."
p str.gsub 'quick','slow'
# "The slow brown fox jumped over the slow dog."

# gsub! will change the varible value, this is the bang utility

# When you run this code, the output is just the sentence without the white spaces 
# before and after the words:

str = ' The quick brown fox jumped over the quick dog.  '
p str.strip

# You'll see that it converts the sentence into an array of words:
# p str.split
