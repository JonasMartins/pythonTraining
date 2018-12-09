from string import Template

def main():
    # Usual string formatting with format()
    str1 = "This is a template {0} by {1}".format("Argument 1", "Argument 2")
    print(str1)

    # create a template with placeholders
    templ = Template("This are some nice language code, ${arg1} and ${arg2}")

    # use the substitute method with keyword arguments
    str2 = templ.substitute(arg1="Python", arg2="Ruby")
    print(str2)

    # use the substitute method with a dictionary
    data = {
        "arg1": "PHP",
        "arg2": "Java"
    }
    str3 = templ.substitute(data)
    print(str3)


if __name__ == "__main__":
    main()
