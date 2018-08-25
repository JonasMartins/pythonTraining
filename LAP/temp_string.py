# demonstrate template string functions

from string import Template

def main():
    # Usual string formatting with format()
    str1 = "You're watching {0} by {1}".format("Berserk", "Kentaro Miura")
    print(str1)
    
    # create a template with placeholders
    templ = Template("You're watching ${title} by ${author}")
    
    # use the substitute method with keyword arguments
    str2 = templ.substitute(title="Berserk", author="Kentaro Miura")
    print(str2)
    
    # use the substitute method with a dictionary
    data = { 
        "author": "Kentaro Miura",
        "title": "Berserk"
    }
    str3 = templ.substitute(data)    
    print(str3)

if __name__ == "__main__":
    main()
    