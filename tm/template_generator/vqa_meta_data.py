from .base import PositionalSynonyms, MetaTemplate

### Please set the "name" of PositionalSynonyms the same as the placeholder in the MetaTemplate
### Have ensured that the first letter of the sentence is capitalized
### Have ensured the generated senetence is striped

# Global PositionalSynonyms
is_line_breaking = PositionalSynonyms("is_line_breaking", ["\n", " "])
is_please = PositionalSynonyms("is_please", [" please ", " "])
is_following = PositionalSynonyms("is_following", [" following ", " "])

# Question MetaTemplates
QuestionMetaTemplates = {
    "Empty":[
        MetaTemplate(
            "{{question}}"
        ),
    ],
    "Declarative":{
        "Simple":{
            "Subject-Verb-Object":[
                MetaTemplate(
                    "The{is_following}question {related_to} the{is_provided}{image} {verb} {object}:{is_line_breaking}{{question}}",
                    [
                        PositionalSynonyms("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        PositionalSynonyms("verb", ["asks for", "requires"]),
                        PositionalSynonyms("object", ["your response", "an answer", "an response", "your answer"]),
                        is_following,
                        is_line_breaking
                    ]
                ),

            ],
            "Subject-LinkingVerb-Complement":[
                # Omit the linking-verb: Question (is)
                MetaTemplate(
                    "Question:{is_line_breaking}{{question}}",
                    [
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "{is_the}question {related_to} the{is_provided}{image}{is}{is_line_breaking}{{question}}",
                    [
                        PositionalSynonyms("is_the", [" ", " the "]),
                        PositionalSynonyms("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        PositionalSynonyms("is", [":", " is:", " is as follows:"]),
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "{intro} is the question:{is_line_breaking}{{question}}",
                    [
                        PositionalSynonyms("intro", ["The following", "Here", "Below", "Presented below", "Presented here"]),
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "{intro} is the question {related_to} the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [
                        PositionalSynonyms("intro", ["The following", "Here", "Below", "Presented below", "Presented here"]),
                        PositionalSynonyms("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
        "Compound":{
            "Joined-By-Coordinating-Conjunctions":[
                MetaTemplate(
                    "The question is {given} {below} {conjunction} you should {answer} it:{is_line_breaking}{{question}}",
                    [
                        PositionalSynonyms("given", ["provided", "given", "presented"]),
                        PositionalSynonyms("below", ["below", "here"]),
                        PositionalSynonyms("conjunction", ["and", "then"]),
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "The question {related_to} the{is_provided}{image} is {given} {below} {conjunction} you should {answer} it:{is_line_breaking}{{question}}",
                    [
                        PositionalSynonyms("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        PositionalSynonyms("given", ["provided", "given", "presented"]),
                        PositionalSynonyms("below", ["below", "here"]),
                        PositionalSynonyms("conjunction", ["and", "then"]),
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking
                    ]
                ),
            ],
            "Joined-By-Semicolons":[
                MetaTemplate(
                    "The question is {given} {below}; you should {answer} it:{is_line_breaking}{{question}}",
                    [
                        PositionalSynonyms("given", ["provided", "given", "presented"]),
                        PositionalSynonyms("below", ["below", "here"]),
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "The question {related_to} the{is_provided}{image} is {given} {below}; you should {answer} it:{is_line_breaking}{{question}}",
                    [
                        PositionalSynonyms("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        PositionalSynonyms("given", ["provided", "given", "presented"]),
                        PositionalSynonyms("below", ["below", "here"]),
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
        "Complex":{
            "Noun-Clauses":[
                MetaTemplate(
                    "The question {given} {below} is what you should {answer}:{is_line_breaking}{{question}}",
                    [

                        PositionalSynonyms("given", ["provided", "given", "presented"]),
                        PositionalSynonyms("below", ["below", "here"]),
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "The question {given} {below} is what you should {answer} {considering}{what_you_see}the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [

                        PositionalSynonyms("given", ["provided", "given", "presented"]),
                        PositionalSynonyms("below", ["below", "here"]),
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        PositionalSynonyms("considering", ["considering", "based on", "referring to"]),
                        PositionalSynonyms("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        is_line_breaking
                    ]
                ),
            ],
            "Adjective-Clauses":[
                MetaTemplate(
                    "The question {which} {adjective}{is_provided}{image} is{is_as_follows}{is_line_breaking}{{question}}",
                    [
                        PositionalSynonyms("which", ["which", "that"]),
                        PositionalSynonyms("adjective", ["is based on the", "is related to the", "you should refer to the", "you should consider the"]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        PositionalSynonyms("is_as_follows",[":", " as follows:"]),
                        is_line_breaking
                    ]
                )
            ],
        },
    },
    "Imperative":{
        "Simple":{
            "Subject-Predicate":[
                MetaTemplate(
                    "{is_please}{answer_directly}:{is_line_breaking}{{question}}",
                    [   
                        PositionalSynonyms("answer_directly", ["answer", "reply", "answer directly", "reply directly"]),
                        is_please,
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "{is_please}{answer_directly} {considering}{what_you_see}the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [   
                        PositionalSynonyms("answer_directly", ["answer", "reply", "answer directly", "reply directly"]),
                        PositionalSynonyms("considering", ["considering", "based on", "referring to"]),
                        PositionalSynonyms("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        is_please,
                        is_line_breaking
                    ]
                )
            ],
            "Subject-Verb-Object":[
                MetaTemplate(
                    "{is_please}{answer} the{is_following}question:{is_line_breaking}{{question}}",
                    [   
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_please,
                        is_following,
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "{is_please}{answer} the{is_following}question {related_to} the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [   
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        PositionalSynonyms("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        is_please,
                        is_following,
                        is_line_breaking
                    ]
                )
            ],
            "Subject-Verb-IndirectObject-DirectObject":[
                MetaTemplate(
                    "{verb} me {the_answer} to the question:{is_line_breaking}{{question}}",
                    [
                        PositionalSynonyms("verb", ["Give", "Provide"]),
                        PositionalSynonyms("the_answer", ["your answer", "the correct answer", "a response"]),
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "{verb} me {the_answer} to the question {related_to} the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [
                        PositionalSynonyms("verb", ["Give", "Provide"]),
                        PositionalSynonyms("the_answer", ["your answer", "the correct answer", "a response"]),
                        PositionalSynonyms("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        is_line_breaking
                    ]
                )
            ]
        },
        "Compound":{
            "Joined-By-Coordinating-Conjunctions":[
                MetaTemplate(
                    "{is_please}{verb}{what_you_see}the{is_provided}{image} and {answer} the{is_following}question:{is_line_breaking}{{question}}", 
                    [
                        PositionalSynonyms("verb", ["analyze", "refer to", "consider"]), 
                        PositionalSynonyms("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),   
            ],
            "Joined-By-Semicolons":[
                MetaTemplate(
                    "{is_please}{verb}{what_you_see}the{is_provided}{image}; {answer} the{is_following}question:{is_line_breaking}{{question}}", 
                    [
                        PositionalSynonyms("verb", ["analyze", "refer to", "consider"]), 
                        PositionalSynonyms("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),  
            ]
        },
        "Complex":{
            "Adverbial-Clauses":[
                MetaTemplate(
                    "{verb}{what_you_see}the{is_provided}{image},{is_please}{answer} the{is_following}question:{is_line_breaking}{{question}}", 
                    [
                        PositionalSynonyms("verb", ["Given", "Analyzing", "Referring to", "Considering"]), 
                        PositionalSynonyms("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),
                MetaTemplate(
                    "{prep}{what_you_see}the{is_provided}{image},{is_please}{answer} the{is_following}question:{is_line_breaking}{{question}}", 
                    [
                        PositionalSynonyms("prep", ["Based on", "From", "According to"]), 
                        PositionalSynonyms("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),
            ],
            "Adjective-Clauses":[
                MetaTemplate(
                    "{is_please}{answer} the question {which} {adjective}{is_provided}{image}:{is_line_breaking}{{question}}",
                    [
                        PositionalSynonyms("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        PositionalSynonyms("which", ["which", "that"]),
                        PositionalSynonyms("adjective", ["is based on the", "is related to the", "you should refer to the", "you should consider the"]),
                        PositionalSynonyms("is_provided", [" ", " provided "]),
                        PositionalSynonyms("image", ["image", "picture"]),
                        is_please,
                        is_line_breaking
                    ]
                )
            ],
        },
    }
}

# Choices MetaTemplates
ChoiceMetaTemplates = {
    "Empty":[
        MetaTemplate(
            "{{choices}}"
        ),
    ],
    "Declarative":{
        "Simple":{
            "Subject-LinkingVerb-Complement":[
                # Omit the linkingVerb: Choices (are)
                MetaTemplate(
                    "{is_the}{is_available}{choices}{are}{is_line_breaking}{{choices}}",
                    [
                        PositionalSynonyms("is_the", [" ", "the"]),
                        PositionalSynonyms("is_available", [" ", " available ", " potential ", " possible "]),
                        PositionalSynonyms("choices", ["choices", "options", "selections"]),
                        PositionalSynonyms("are", [":", " are:"]),
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "{is_the}{is_available}{choices} are as follows:{is_line_breaking}{{choices}}",
                    [
                        PositionalSynonyms("is_the", [" ", "the"]),
                        PositionalSynonyms("is_available", [" ", " available ", " potential ", " possible "]),
                        PositionalSynonyms("choices", ["choices", "options", "selections"]),
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "{is_the}{is_available}{choices} are {provided}{below}{is_line_breaking}{{choices}}",
                    [
                        PositionalSynonyms("is_the", [" ", "the"]),
                        PositionalSynonyms("is_available", [" ", " available ", " potential ", " possible "]),
                        PositionalSynonyms("choices", ["choices", "options", "selections"]),
                        PositionalSynonyms("provided", ["provided", "offered", "presented", "listed"]),
                        PositionalSynonyms("below", [":", " below:", " here:", " as follows:"]),
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "{adv} are the{is_available}{choices}:{is_line_breaking}{{choices}}",
                    [
                        PositionalSynonyms("adv", ["The following", "Here", "Below", "Presented below", "Presented here"]),
                        PositionalSynonyms("is_available", [" ", " available ", " potential ", " possible "]),
                        PositionalSynonyms("choices", ["choices", "options", "selections"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
        "Compound":{
            "Joined-By-Coordinating-Conjunctions":[
                MetaTemplate(
                    "{is_the}{is_available}{choices} are {provided} {below} {conjunction} you should {verb} {object}:{is_line_breaking}{{choices}}",
                    [
                        PositionalSynonyms("is_the", [" ", "the"]),
                        PositionalSynonyms("is_available", [" ", " available ", " potential ", " possible "]),
                        PositionalSynonyms("choices", ["choices", "options", "selections"]),
                        PositionalSynonyms("provided", ["provided", "offered", "presented", "listed"]),
                        PositionalSynonyms("below", ["below", "here", "as follows"]),
                        PositionalSynonyms("conjunction", ["and", "then"]),
                        PositionalSynonyms("verb", ["select", "choose", "pick"]),
                        PositionalSynonyms("object", ["the best one", "the best fit", "the best answer"]),
                        is_line_breaking
                    ]
                ),
            ],
            "Joined-By-Semicolons":[
                MetaTemplate(
                    "{is_the}{is_available}{choices} are {provided} {below}; you should {verb} {object}:{is_line_breaking}{{choices}}",
                    [
                        PositionalSynonyms("is_the", [" ", "the"]),
                        PositionalSynonyms("is_available", [" ", " available ", " potential ", " possible "]),
                        PositionalSynonyms("choices", ["choices", "options", "selections"]),
                        PositionalSynonyms("provided", ["provided", "offered", "presented", "listed"]),
                        PositionalSynonyms("below", ["below", "here", "as follows"]),
                        PositionalSynonyms("verb", ["select", "choose", "pick"]),
                        PositionalSynonyms("object", ["the best one", "the best fit", "the best answer"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
        "Complex":{
            "Adjective-Clauses":[
                MetaTemplate(
                    "{is_the}{is_available}{choices} {which} include only one {correct} answer{are}{is_line_breaking}{{choices}}",
                    [
                        PositionalSynonyms("is_the", [" ", "the"]),
                        PositionalSynonyms("is_available", [" ", " available ", " potential ", " possible "]),
                        PositionalSynonyms("choices", ["choices", "options", "selections"]),
                        PositionalSynonyms("which", ["which", "that"]),
                        PositionalSynonyms("correct", ["correct", "best", "most suitable"]),
                        PositionalSynonyms("are", [":", " are:"]),
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "{is_the}{is_available}{choices} {which} include only one {correct} answer are as follows:{is_line_breaking}{{choices}}",
                    [
                        PositionalSynonyms("is_the", [" ", "the"]),
                        PositionalSynonyms("is_available", [" ", " available ", " potential ", " possible "]),
                        PositionalSynonyms("choices", ["choices", "options", "selections"]),
                        PositionalSynonyms("which", ["which", "that"]),
                        PositionalSynonyms("correct", ["correct", "best", "most suitable"]),
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "{is_the}{is_available}{choices} {which} include only one {correct} answer are {provided}{below}{is_line_breaking}{{choices}}",
                    [
                        PositionalSynonyms("is_the", [" ", "the"]),
                        PositionalSynonyms("is_available", [" ", " available ", " potential ", " possible "]),
                        PositionalSynonyms("choices", ["choices", "options", "selections"]),
                        PositionalSynonyms("which", ["which", "that"]),
                        PositionalSynonyms("correct", ["correct", "best", "most suitable"]),
                        PositionalSynonyms("provided", ["provided", "offered", "presented", "listed"]),
                        PositionalSynonyms("below", [":", " below:", " here:", " as follows:"]),
                        is_line_breaking
                    ]
                ),
                MetaTemplate(
                    "{adv} are the{is_available}{choices} {which} include only one {correct} answer:{is_line_breaking}{{choices}}",
                    [
                        PositionalSynonyms("adv", ["The following", "Here", "Below", "Presented below", "Presented here"]),
                        PositionalSynonyms("is_available", [" ", " available ", " potential ", " possible "]),
                        PositionalSynonyms("choices", ["choices", "options", "selections"]),
                        PositionalSynonyms("which", ["which", "that"]),
                        PositionalSynonyms("correct", ["correct", "best", "most suitable"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
    },
    "Imperative":{
        "Simple":{
            "Subject-Predicate":[
                MetaTemplate(
                    "{is_please}{verb} from:{choice}{{choices}}",
                    [
                        PositionalSynonyms("verb", ["pick", "select", "choose"]),
                        PositionalSynonyms("choice", ["\nChoices: ", "\nOptions: ", "\n"]),
                        is_please
                    ]
                )
            ],
            "Subject-Verb-Object":[
                # Omit the subject
                MetaTemplate(
                    "{is_please}{verb} the{adj}{answer}{from}{to} the question.{choice}{{choices}}",
                    [
                        PositionalSynonyms("verb", ["make", "pick", "indicate", "select", "choose"]),
                        PositionalSynonyms("adj", [" right ", " correct ", " accurate ", " "]),
                        PositionalSynonyms("answer", ["answer", "choice", "option", "response", "solution"]),
                        PositionalSynonyms("from", [
                            " from the provided options ", " from the options given ",
                            " from the provided choices ", " from the choices given ", 
                            " from the choices below ", " from the options below ", 
                            " from the available choices ", " from the available options ", " "
                        ]),
                        PositionalSynonyms("to", [
                            "to correctly answer", "to answer", 
                            "to correctly address", "to address", 
                            "to correctly respond to", "to respond to"
                        ]),
                        PositionalSynonyms("choice", ["\nChoices: ", "\nOptions: ", "\n"]),
                        is_please
                    ]
                )
            ],
        },
        "Complex":{
            "Adjective-Clauses":[
                MetaTemplate(
                    "{is_please}{verb} the{adj}{answer}{from}{which} include only one {correct} answer {to} the question.{choice}{{choices}}",
                    [
                        PositionalSynonyms("verb", ["make", "pick", "indicate", "select", "choose"]),
                        PositionalSynonyms("adj", [" right ", " correct ", " accurate ", " "]),
                        PositionalSynonyms("answer", ["answer", "choice", "option", "response", "solution"]),
                        PositionalSynonyms("from", [
                            " from the provided options ", " from the options given ",
                            " from the provided choices ", " from the choices given ", 
                            " from the choices below ", " from the options below ", 
                            " from the available choices ", " from the available options ", " "
                        ]),
                        PositionalSynonyms("which", ["which", "that"]),
                        PositionalSynonyms("correct", ["correct", "best", "most suitable"]),
                        PositionalSynonyms("to", [
                            "to correctly answer", "to answer", 
                            "to correctly address", "to address", 
                            "to correctly respond to", "to respond to"
                        ]),
                        PositionalSynonyms("choice", ["\nChoices: ", "\nOptions: ", "\n"]),
                        is_please
                    ]
                )
            ],
        },
    }
}