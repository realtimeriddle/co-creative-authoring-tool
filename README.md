# co-creative-authoring-tool
A Hybrid Approach to Co-Creative Story Authoring Using Grammars and Language Models

# Instructions
1. install pytorch
2. install requriements
3. download models and place them in coAuthoringToolPrototype\models folder
4. run main.py. Use the paramiter --no_cuda to force models to run without cuda.

# Pretrained-Models
Download: https://drive.google.com/drive/folders/1k_soPZWP6ryfCDcszCzpwzVa03mqnd-0?usp=sharing

# Video Demo
https://youtu.be/pUbJb09bzCQ

# Workflow Explanation
## 1) Gpt2 Window
This window allows the user to use Gpt2 to generate what we call "unprocessed" text. Unprocessed text is composed of phrases that the user would like to appear in the final text in the order they are written, separated by the special separation token ```<|sepofcond|>```

The Generate button triggers Gpt2 to generate text using the settings in the Controls panel. The <|sepofocond|> button prints the ```<|sepofcond|>``` to avoid mistyping the important seperation token.

## 2) Tracery Window
The Tracery window lets the user create and run a tracery gramar. For a information on creating tracery grammars, see Tracery's interactive tutorial at https://www.tracery.io/

The Tracery Window again provides a ```<|sepofcond|>``` button as well a Run button that prints the output of the grammar to the editor window starting at the list name written in the origin text box next to the run button.

## 3) Classifier Window
Sometimes when writing a grammar quickly creating lists can useful. The classifier window aids in this task by shortening lists using a zero-shot classifier. 

Lists are stored in txt files in the *Lists* folder of the repository and can be added directly to the folder or pasted into the provided textbox in the upper left corner of the window. Saved lists can be selected in the list box in the upper right corner.

Labels for classification can be added and removed with the *+* and *-* buttons. Choose what labled results you want in the final list by selecting the check box next to the desired labels. If items in your list can have multiple labels choose the multilabel selection box.

Adding a hypothesis template lets the model see the labels in a context instead of stand alone words. A hypothesis template is created by typeing a sentence into the Hypothesis Template textbox including ```{}``` where the label would be placed while classifying.
Ex. ```This item is an example of {}``` or ```The sentiment of the text is {}```

The Threshold slider configures the probability the result must reach inorder to be included in the new list.

Lists can be saved by typing the name of the list into the text box at the bottom of the window and pressing the save list button.

Pressing the *Run Classifier* button runs the zeroshot classifier and only items with the selected will remain in the list in the upper left corner when classification is compleated.

## 4) The Editor Window
The Editor Window is used to "process" the "unprocessed" grammar output.

The user can proccess text by highlighting the text they would like to process and selecting the ```Process``` button. The processed text will then be printed at the location of the cursor. The user can then continue to edit and process text until compleation.

A  ```<|sepofcond|>``` button is also provided to make adding separation tokens easier.

WARNING: If input text is too long for the models an ERROR will be printed instead. If you see this, try processing text in smaller chunks.

