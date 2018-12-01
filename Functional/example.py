# 2 elevado a um numero x -1, a potencia precede a subtracao
functional = lambda x: 2**x-1

# print(functional(5))

# High order functions
year_cheese = [(2000, 29.87), (2001, 30.12), (2002, 30.6), (2003,  30.66),(2004, 31.33), (2005, 32.62), (2006, 32.73), (2007, 33.5),  (2008, 32.84), (2009, 33.02), (2010, 32.92)]

# print(max(year_cheese))

t = max(map(lambda yc: (yc[1],yc), year_cheese))

print(t[1])

snd= lambda x: x[1] 
snd( max(map(lambda yc: (yc[1],yc), year_cheese)))
