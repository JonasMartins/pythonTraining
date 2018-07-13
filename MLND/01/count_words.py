""" Count words """

def count_words(s,n):
	""" Return the most frequently occuring words in s. """
	# list = [123, 'xyz', 'zara', 'abc', 'xyz']
	word = 'cat'
	# top_n = s.count(word)
	# top_n = s.split()
	# top_n.sort()
	# list.sort()
	# top_n = list

	# get the instace class name .__class__.__name__

	# TODO: Count the occurences of each workd in s
	top_n = s.split()
	occurences = list()
	non_duplicates = list(set(top_n)) # a list with all words non duplicated
	for i in non_duplicates:
		occurences.append(top_n.count(i))

	# TODO: Sort the occurences in descending order (alphabetically in case of ties)
	occurences.sort(reverse=True)

	# occurences have the number of occurences of each respective word on the given list
	print occurences



	return top_n

def test_run():
	""" Test count_words with some inputs """
	print count_words("cat bar mat cat bat cat",3)
	# print count_words("betty bought a bit of butter but the butter was bitter",3)

	# ...... forget this

if __name__ == '__main__':
	test_run()