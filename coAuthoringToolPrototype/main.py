import PySimpleGUI as sg
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM,  BartTokenizer, BartForConditionalGeneration
from os import listdir
from os.path import join, isfile
import tracery
from tracery.modifiers import base_english
import json
import tkinter as tk
import torch

def run_classifier(inputText, candidate_labels, outputLabels, hypothesis_template, threshold, multilabel):

    outputText = ''

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli", low_cpu_mem_usage=True, device=torch.cuda.current_device())
    else:
        classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli", low_cpu_mem_usage=True)

    if "{}" in hypothesis_template:
        output = classifier(inputText, candidate_labels, hypothesis_template=hypothesis_template, multilabel=multilabel)

    else:
        output = classifier(inputText, candidate_labels, multilabel=multilabel)

    for seq in output:
        if seq['scores'][0] >= threshold and seq['labels'][0] in outputLabels:
            outputText = outputText + seq['sequence'] + '\n'

    return outputText[:-1]


def run_generator(inText):

    tokenizer = AutoTokenizer.from_pretrained("models/gpt2_stage0")

    model = AutoModelForCausalLM.from_pretrained("models/gpt2_stage0", low_cpu_mem_usage=True)

    encoded_text = tokenizer.encode('<|STORY|>'+inText, add_special_tokens=False, return_tensors="pt")

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        model = model.to(torch.cuda.current_device())
        encoded_text = encoded_text.to(torch.cuda.current_device())

    generated_text = model.generate(input_ids=encoded_text, num_beams=2, do_sample=True, top_k=25, top_p=1.0,
                                    max_length=100 + len(encoded_text[0]), min_length=100 + len(encoded_text[0]),
                                    repetition_penalty=1.3, no_repeat_ngram_size=3, temperature=1.0)

    if len(generated_text.shape) > 2:
        generated_text.squeeze_()

    outText = tokenizer.decode(generated_text[0])

    return outText

def run_progressive_generation(inputStr):
    tokenizer1 = BartTokenizer.from_pretrained('models/bart_stage1')
    model1 = BartForConditionalGeneration.from_pretrained('models/bart_stage1', low_cpu_mem_usage=True)

    tokenizer2 = BartTokenizer.from_pretrained('models/bart_stage2')
    model2 = BartForConditionalGeneration.from_pretrained('models/bart_stage2', low_cpu_mem_usage=True)

    inputs1 = tokenizer1.batch_encode_plus(['<|startofcond|>'+inputStr], return_tensors='pt')

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        model1 = model1.to(torch.cuda.current_device())
        inputs1 = inputs1.to(torch.cuda.current_device())

    output_ids = model1.generate(inputs1['input_ids'], num_beams=2, early_stopping=True)

    text = tokenizer1.decode(output_ids[0], clean_up_tokenization_spaces=True).replace('</s>', '').replace('<s>', '')

    inputs2 = tokenizer2.batch_encode_plus([text], return_tensors='pt')

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        model2 = model2.to(torch.cuda.current_device())
        inputs2 = inputs2.to(torch.cuda.current_device())

    output_ids = model2.generate(inputs2['input_ids'], num_beams=2, early_stopping=True)

    outStr = tokenizer2.decode(output_ids[0], clean_up_tokenization_spaces=True).replace('</s>', '').replace('<s>', '')

    return outStr



def make_classifier_window():
    classifier_Layout = [[sg.Multiline('', autoscroll=True, key='Classifier Output', expand_x=True, expand_y=True, disabled=False), sg.Listbox([], expand_x=True, expand_y=True, enable_events=True, key="Classifier ListBox")],
                         [sg.Column([[sg.Checkbox('', key='Label1Box'), sg.Input(expand_x=True, key='Label1', expand_y=True)]], vertical_scroll_only=True, scrollable=True, expand_y=True, expand_x=True, size=(None, 100), key='Label Col'), sg.B('+', key='Add Label'), sg.B('-', key='Remove Label')],
                          [sg.Text("Hypothesis Template:"), sg.Input(expand_x=True, key='Classifier Template'), sg.Checkbox("MultiLabel:", key='MultiLabel Box') ],
                         [sg.Text("Threshold:"), sg.Slider(range=(0.00, 1.0), default_value=0.50, expand_x=True, expand_y=True, orientation='h', resolution=0.01, key='Threshold Slider')],
                    [sg.Button('Run Classifier'), sg.Input(expand_x=True, key='Classifier File Name'), sg.Button('Save List')]]

    return sg.Window('Classifier', classifier_Layout, resizable=True, finalize=True, auto_size_text=True)

def make_generator_window():

    generatorLayout = [[sg.Multiline('', autoscroll=True, key='Generator', expand_x=True, expand_y=True)],
                    [sg.Button('Generate'), sg.Button('<|sepofcond|>')]]

    return sg.Window('Gpt2', generatorLayout, resizable=True, finalize=True, auto_size_text=True)

def make_tracery_window():
    traceryLayout = [[sg.Multiline('{\n\t"origin\": []\n}', autoscroll=True, key='Tracery Editor', expand_x=True, expand_y=True)],
                    [sg.Button('<|sepofcond|>'), sg.Button('Run'), sg.Text("origin:"), sg.Input(expand_x=True, key='Grammar Origin')]]

    return sg.Window('Tracery', traceryLayout, resizable=True, finalize=True, auto_size_text=True)

def make_editor_window():
    editorLayout = [[sg.Multiline('', autoscroll=True, key='Text Editor', expand_x=True, expand_y=True)],
                    [sg.Button('Process'), sg.Button('<|sepofcond|>')]]

    return sg.Window('Editor', editorLayout, resizable=True, finalize=True, auto_size_text=True)





sg.theme('DarkBlue')   # Add a touch of color

# Variables

numAttrib = 1
numChar = 1
numLabels = 1

attributesTypes = ['text', 'combo', 'bool', "list", "slider"]

storyStructure = []

classifierWindow = make_classifier_window()
generatorWindow = make_generator_window()
traceryWindow = make_tracery_window()
editorWindow = make_editor_window()

listDir = 'Lists'

# Main loop
while True:

# read events

    window, event, values = sg.read_all_windows(timeout=1)

    classifierWindow['Label Col'].expand(False, True)
    classifierWindow['Classifier Template'].expand(True, False)


    if event == sg.WIN_CLOSED:
        break

# Classifier Events

    # update list Files

    listFiles = []
    for f in listdir(listDir):
        if isfile(join(listDir, f)):
            listFiles.append(f)

    classifierWindow["Classifier ListBox"].update(values=listFiles)


    if window == classifierWindow and event == 'Run Classifier':

        labels = []
        outLabels = []
        inValues = []
        hTemplate = ''
        threshold = 0
        mLabel = False

        for l in range(1, numLabels+1):
            if values['Label'+str(l)] != '':
                labels.append(values['Label'+str(l)])
                if values['Label'+str(l)+'Box']:
                    outLabels.append(values['Label' + str(l)])

        inValues = values["Classifier Output"].split('\n')
        hTemplate = values['Classifier Template']
        tHold = values['Threshold Slider']
        mLabel = values['MultiLabel Box']



        if values['Classifier Output'] != '' and len(labels) and len(inValues):
            classifierWindow["Classifier Output"].update(run_classifier(inValues, labels, outLabels, hTemplate, tHold, mLabel))


    elif window == classifierWindow and event == 'Add Label':
        numLabels = numLabels + 1

        if 'Label' + str(numLabels) in classifierWindow.key_dict:
            classifierWindow['Label' + str(numLabels)].unhide_row()

        else:
            classifierWindow.extend_layout(classifierWindow['Label Col'], [[sg.Checkbox('', key='Label' + str(numLabels) + 'Box'),sg.Input(expand_x=True, key='Label' + str(numLabels))]])

        classifierWindow['Label Col'].contents_changed()

    elif window == classifierWindow and event == 'Remove Label':
        if numLabels > 1:
            classifierWindow['Label' + str(numLabels)].hide_row()
            classifierWindow['Label Col'].contents_changed()
            numLabels = numLabels - 1

    elif window == classifierWindow and event == "Classifier ListBox" and len(values['Classifier ListBox']):
        infile = open(join(listDir, values['Classifier ListBox'][0]), "r")

        classifierWindow["Classifier Output"].update(infile.read())

        infile.close()



    elif window == classifierWindow and event == "Save List":

        outfile = open(join(listDir, values['Classifier File Name']+".txt"), "w")
        outfile.write(values["Classifier Output"])
        outfile.close()


# Generator Events

    if window == generatorWindow and event == 'Generate':

        generatorWindow["Generator"].update(run_generator(values["Generator"]))

    elif window == generatorWindow and event == '<|sepofcond|>':
        generatorWindow['Generator'].Widget.insert(generatorWindow['Generator'].Widget.index(tk.INSERT), '<|sepofcond|>')

# Tracery Events

    if window == traceryWindow and event == 'Run':

        try:
            grammar = tracery.Grammar(json.loads(values['Tracery Editor']))
            grammar.add_modifiers(base_english)
            editorWindow['Text Editor'].Widget.insert(editorWindow['Text Editor'].Widget.index(tk.INSERT),
                                                      grammar.flatten("#"+values['Grammar Origin']+"#"))

        except:
            editorWindow['Text Editor'].Widget.insert(editorWindow['Text Editor'].Widget.index(tk.INSERT),
                                                      'ERROR')
    elif window == traceryWindow and event == '<|sepofcond|>':
        traceryWindow['Tracery Editor'].Widget.insert(traceryWindow['Tracery Editor'].Widget.index(tk.INSERT), '<|sepofcond|>')



# Text editor Events

    if window == editorWindow and event == 'Process':

        try:
            editorWindow['Text Editor'].Widget.insert(editorWindow['Text Editor'].Widget.index(tk.INSERT),
                                                  run_progressive_generation(editorWindow['Text Editor'].Widget.selection_get()))
        except:
            editorWindow['Text Editor'].Widget.insert(editorWindow['Text Editor'].Widget.index(tk.INSERT),
                                                      'ERROR')

    elif window == editorWindow and event == '<|sepofcond|>':
        editorWindow['Text Editor'].Widget.insert(editorWindow['Text Editor'].Widget.index(tk.INSERT), '<|sepofcond|>')

