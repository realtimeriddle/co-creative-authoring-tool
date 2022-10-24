import PySimpleGUI as sg
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM,  BartTokenizer, BartForConditionalGeneration
from os import listdir
from os.path import join, isfile
import tracery
from tracery.modifiers import base_english
import json
import tkinter as tk
import torch
import argparse


def run_classifier(inputText, candidate_labels, outputLabels, hypothesis_template, threshold, multilabel, args):
    # Runs Zero-shot Classification

    outputText = ''

    if torch.cuda.is_available() and torch.cuda.device_count() > 0 and not args.no_cuda:
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


def run_generator(inText, args, num_beams, do_sample, top_k, top_p, length, repetition_penalty, no_repeat_ngram_size, temperature):
    # Runs Causal Text Generation

    ENCODED_MAX = 750

    tokenizer = AutoTokenizer.from_pretrained("models/gpt2_stage0")

    model = AutoModelForCausalLM.from_pretrained("models/gpt2_stage0", low_cpu_mem_usage=True)

    if inText == '':
        inText = '<|endoftext|>'


    temp_encoded = tokenizer.encode(inText)


    if len(temp_encoded) > ENCODED_MAX:
        temp_encoded = temp_encoded[-ENCODED_MAX:]

    if len(temp_encoded) + length > 1024:
        length = 1024 - len(temp_encoded)

    inText = tokenizer.decode(temp_encoded)

    encoded_text = tokenizer.encode(inText, add_special_tokens=False, return_tensors="pt")

    if torch.cuda.is_available() and torch.cuda.device_count() > 0 and not args.no_cuda:
        model = model.to(torch.cuda.current_device())
        encoded_text = encoded_text.to(torch.cuda.current_device())

    generated_text = model.generate(input_ids=encoded_text, num_beams=int(num_beams), do_sample=do_sample, top_k=int(top_k), top_p=top_p,
                                    max_length=int(length) + len(encoded_text[0]), min_length=int(length) + len(encoded_text[0]),
                                    repetition_penalty=repetition_penalty, no_repeat_ngram_size=int(no_repeat_ngram_size), temperature=temperature)

    if len(generated_text.shape) > 2:
        generated_text.squeeze_()

    prompt_length = len(tokenizer.encode(inText))

    outText = tokenizer.decode(generated_text[0][prompt_length:])

    return outText

def run_progressive_generation(inputStr, args):
    # Runs MultiStage Text to Text Generation

    ENCODED_MAX = 1024

    tokenizer1 = BartTokenizer.from_pretrained('models/bart_stage1')
    model1 = BartForConditionalGeneration.from_pretrained('models/bart_stage1', low_cpu_mem_usage=True)

    tokenizer2 = BartTokenizer.from_pretrained('models/bart_stage2')
    model2 = BartForConditionalGeneration.from_pretrained('models/bart_stage2', low_cpu_mem_usage=True)

    if torch.cuda.is_available() and torch.cuda.device_count() > 0 and not args.no_cuda:
        model1.to('cuda')
        model2.to('cuda')

    # encoding the prompt first just to check the length
    temp_encoded = tokenizer1.encode(inputStr, add_special_tokens=False)

    #check if string is too long for model
    if len(temp_encoded) <= ENCODED_MAX:

        # first stage

        inputs1 = tokenizer1.batch_encode_plus([inputStr], return_tensors='pt')

        if torch.cuda.is_available() and torch.cuda.device_count() > 0 and not args.no_cuda:
            inputs1 = inputs1.to('cuda')

        output_ids = model1.generate(inputs1['input_ids'], num_beams=4, early_stopping=True, max_length=1024)

        text = tokenizer1.decode(output_ids[0], clean_up_tokenization_spaces=True).replace('</s>', '').replace(
            '<s>',
            '')

        # second stage

        inputs2 = tokenizer2.batch_encode_plus([text], return_tensors='pt')

        if torch.cuda.is_available() and torch.cuda.device_count() > 0 and not args.no_cuda:
            inputs2 = inputs2.to('cuda')

        output_ids = model2.generate(inputs2['input_ids'], num_beams=4, early_stopping=True, max_length=1024)

        outStr = tokenizer2.decode(output_ids[0], clean_up_tokenization_spaces=True).replace('</s>', '').replace(
            '<s>',
            '')

    else:
        #if string is too long return original Text and print error
        outStr = inputStr
        print("Error: Text length", len(temp_encoded))

    return outStr


########## Functions that build window layouts for PySimpleGUI

def make_classifier_window():
    classifier_Layout = [[sg.Multiline('', autoscroll=True, key='Classifier Output', expand_x=True, expand_y=True, disabled=False), sg.Listbox([], expand_x=True, expand_y=True, enable_events=True, key="Classifier ListBox")],
                         [sg.Column([[sg.Checkbox('', key='Label1Box'), sg.Input(expand_x=True, key='Label1', expand_y=True)]], vertical_scroll_only=True, scrollable=True, expand_y=True, expand_x=True, size=(None, 100), key='Label Col'), sg.B('+', key='Add Label'), sg.B('-', key='Remove Label')],
                          [sg.Text("Hypothesis Template:"), sg.Input(expand_x=True, key='Classifier Template'), sg.Checkbox("MultiLabel:", key='MultiLabel Box') ],
                         [sg.Text("Threshold:"), sg.Slider(range=(0.00, 1.0), default_value=0.50, expand_x=True, expand_y=True, orientation='h', resolution=0.01, key='Threshold Slider')],
                    [sg.Button('Run Classifier'), sg.Input(expand_x=True, key='Classifier File Name'), sg.Button('Save List')]]

    return sg.Window('Classifier', classifier_Layout, resizable=True, finalize=True, auto_size_text=True)

def make_generator_window():

    generatorLayout = [[sg.Multiline('', autoscroll=True, key='Generator', expand_x=True, expand_y=True)],
                    [sg.Button('Generate'), sg.Button('<|sepofcond|>'), sg.Frame('Controls', [[sg.Text("num_beams:"), sg.Slider(range=(1, 10), default_value=1, expand_x=True, expand_y=True, orientation='h', resolution=1, key='Beams Slider')],
                                                                                              [sg.Checkbox("do_sample:", key='Sample Box')],
                                                                                              [sg.Text("top_k:"), sg.Slider(range=(1, 100), default_value=1, expand_x=True, expand_y=True, orientation='h', resolution=1, key='TopK Slider')],
                                                                                              [sg.Text("top_p:"), sg.Slider(range=(0.00, 1.00), default_value=1.00, expand_x=True, expand_y=True, orientation='h', resolution=0.01, key='TopP Slider')],
                                                                                              [sg.Text("length:"), sg.Slider(range=(1, 500), default_value=1, expand_x=True, expand_y=True, orientation='h', resolution=1, key='Length Slider')],
                                                                                              [sg.Text("repetition penalty:"), sg.Slider(range=(1.0, 2.0), default_value=1.00, expand_x=True, expand_y=True, orientation='h', resolution=0.1, key='RP Slider')],
                                                                                              [sg.Text("no_repeat_ngram:"), sg.Slider(range=(1, 10), default_value=1, expand_x=True, expand_y=True, orientation='h', resolution=1, key='Ngram Slider')],
                                                                                              [sg.Text("temp:"), sg.Slider(range=(0.00, 2.00), default_value=1.00, expand_x=True, expand_y=True, orientation='h', resolution=0.01, key='temp Slider')]])]]

    return sg.Window('Gpt2', generatorLayout, resizable=True, finalize=True, auto_size_text=True)

def make_tracery_window():
    traceryLayout = [[sg.Multiline('{\n\t"origin\": []\n}', autoscroll=True, key='Tracery Editor', expand_x=True, expand_y=True)],
                    [sg.Button('<|sepofcond|>'), sg.Button('Run'), sg.Text("origin:"),
                     sg.Input(expand_x=True, key='Grammar Origin'), sg.Text('Load'),
                     sg.Combo([], key='TraceryCombo', expand_x=True, enable_events=True), sg.Button('Save'),
                     sg.Input(expand_x=True, key='SaveGrammarName')]]

    return sg.Window('Tracery', traceryLayout, resizable=True, finalize=True, auto_size_text=True)

def make_editor_window():
    editorLayout = [[sg.Multiline('', autoscroll=True, key='Text Editor', expand_x=True, expand_y=True)],
                    [sg.Button('Process'), sg.Button('<|sepofcond|>')]]

    return sg.Window('Editor', editorLayout, resizable=True, finalize=True, auto_size_text=True)


##########

######### Checks for arguments
# taken from the transformers library
parser = argparse.ArgumentParser()

parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
args = parser.parse_args()

###########

##########PySimpleGUI setup
sg.theme('DarkBlue')   # Add a touch of color

classifierWindow = make_classifier_window()
generatorWindow = make_generator_window()
traceryWindow = make_tracery_window()
editorWindow = make_editor_window()
##########

# Folder location for the lists used by the classifier
listDir = 'Lists'

# Folder location for the grammars used by the tracery window
grammarDir = 'Grammars'

# PySimpleGUI Main loop
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
            classifierWindow["Classifier Output"].update(run_classifier(inValues, labels, outLabels, hTemplate, tHold, mLabel, args))


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

        generatorWindow['Generator'].Widget.insert(generatorWindow['Generator'].Widget.index(tk.INSERT),
                                                   run_generator(generatorWindow['Generator'].Widget.get("1.0", generatorWindow['Generator'].Widget.index(tk.INSERT)), args, values["Beams Slider"], values["Sample Box"], values["TopK Slider"], values["TopP Slider"], values["Length Slider"], values["RP Slider"], values["Ngram Slider"], values["temp Slider"]))

    elif window == generatorWindow and event == '<|sepofcond|>':
        generatorWindow['Generator'].Widget.insert(generatorWindow['Generator'].Widget.index(tk.INSERT), '<|sepofcond|>')

# Tracery Events

    #update save grammar list

    grammarFiles = []
    for f in listdir(grammarDir):
        if isfile(join(grammarDir, f)):
            grammarFiles.append(f)

    traceryWindow["TraceryCombo"].update(values=grammarFiles)


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

    elif window == traceryWindow and event == 'Save':
        outfile = open(join(grammarDir, values['SaveGrammarName'] + ".json"), "w")
        outfile.write(values["Tracery Editor"])
        outfile.close()


    elif window == traceryWindow and event == 'TraceryCombo' and len(values['TraceryCombo']):

        f = open(grammarDir+'\\'+values['TraceryCombo'])

        traceryWindow['Tracery Editor'].update(f.read())

        f.close()

# Text editor Events

    if window == editorWindow and event == 'Process':

        try:
            editorWindow['Text Editor'].Widget.insert(editorWindow['Text Editor'].Widget.index(tk.INSERT),
                                                  run_progressive_generation(editorWindow['Text Editor'].Widget.selection_get(), args))
        except:
            editorWindow['Text Editor'].Widget.insert(editorWindow['Text Editor'].Widget.index(tk.INSERT),
                                                      'ERROR')

    elif window == editorWindow and event == '<|sepofcond|>':
        editorWindow['Text Editor'].Widget.insert(editorWindow['Text Editor'].Widget.index(tk.INSERT), '<|sepofcond|>')

