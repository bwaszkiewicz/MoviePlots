import pandas

word = "Alamam[12]kota"
pocz = word.find('[')
kon = word.find(']')

print ("Poczatek ", pocz)
print ("Koniec ", kon)

newWord = word[:pocz] + word[kon+1:]
print(newWord)
#print(word.translate())