
import os
import numpy as np

from bs4 import BeautifulSoup
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

xrays_dir = "../../openi-chest-xrays/data/xrays" #from VAE/data_process file.
reports_dir = "../../openi-chest-xrays/data/xrays"
show_histogram = True

def extract_label_from_report(parsed_report):
    '''
    parsed_report (bs4.BeautifulSoup): output of parse_report()
    
    The "normal" label does not overlap with any other label.
    So, if a mesh major is "normal", we return 0. Else, return 1.
    '''
    ''' majors = parsed_report.mesh.findAll("major")
    for major in majors:
        if major.getText() == "normal":
            return 0
    return 1 '''
    
    label_info = [0, "normal"]
    majors = parsed_report.mesh.findAll("major")
    for major in majors:
        descr = major.getText()

        slash_idx = descr.find('/')
        if slash_idx > -1:
            descr = descr[:slash_idx]
        
        comma_idx = descr.find(',')
        if comma_idx > -1:
            descr = descr[:comma_idx]
        
        if descr == "No Indexing":
            return -1
        elif descr in ["Calcinosis", "Opacity", "Thoracic Vertebrae", "Calcified Granuloma"]:
            label_info = [1, descr]
        else:
            label_info = [0, descr]

    return label_info

def extract_xray_paths_from_report(parsed_report):
    '''
    parsed_report (bs4.BeautifulSoup): output of parse_report()
    '''
    xrays = parsed_report.findAll("parentimage")
    xray_paths = []
    for xray in xrays:
        xray_path = os.path.join(xrays_dir, xray["id"] + ".png")
        xray_path = os.path.abspath(xray_path)
        xray_paths.append(xray_path)
    return xray_paths

def parse_report(report):
    '''
    report (string): filepath eg. "<path_to_raw_reports>/1.xml"
    '''
    f = open(report, 'r')
    report_xml = f.readlines()
    report_xml = ''.join(report_xml)
    report_parsed = BeautifulSoup(report_xml, "lxml")
    f.close()
    return report_parsed

def build_dataset(reports):
    '''
    reports (list): list of pathnames eg. ["<path_to_raw_reports>/1.xml", ...]
    '''
    dataset = []
    for report in tqdm(reports):
        parsed = parse_report(report)
        xray_paths = extract_xray_paths_from_report(parsed)
        if len(xray_paths) == 0: continue
        label_info = extract_label_from_report(parsed)
        if label_info == -1:
            continue
        entry = { "xray_paths" : xray_paths, 
                 "label" : label_info[0],
                 "description" : label_info[1]}
        if entry["label"] == 0:
            #in distribution, so save img at xray_paths to PetImages/train/xray/***.jpg
            shutil.copy(entry["xray_paths"][0], "../../InDistribution")
        else:
            shutil.copy(entry["xray_paths"][0], "../../OutDistribution")

def numerical_sort_key(report):
    '''
    report (string): filename eg. "1.xml"
    '''
    return int(report.split(".")[0])

def get_all_reports():
    '''
    returns a list of pathnames to reports
    '''
    reports = os.listdir(reports_dir)
    reports.sort(key=numerical_sort_key) 
    reports = [os.path.abspath(os.path.join(reports_dir, report)) \
                 for report in reports]
    return reports

if __name__ == "__main__":
    reports = get_all_reports()
    dataset = build_dataset(reports)