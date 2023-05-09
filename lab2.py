'''
File: lab2.py
Author: Derrick Kempster
Purpose: Program for lab 2
Usage: python3 lab2.py train <examples> <hypothesis> <learning-type>
       python3 lab2.py predict <hypothesis> <file>
Description of program functionality is in the write-up for this project.
'''

import math
import sys

# feature constants
CAPSALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DUTCHEXTRA = 'ÄËÉÈÏĲÖÜ'
HMAX = 104
MAXDEPTH = 10

# language and method strings
EN = 'en'
NL = 'nl'
DT = 'dt'
ADA = 'ada'

# threshholds to determine feature values
ATHRESH = 0.08015
BTHRESH = 0.01470
CTHRESH = 0.02010
DTHRESH = 0.04775
ETHRESH = 0.15850
FTHRESH = 0.01380
GTHRESH = 0.02525
HTHRESH = 0.04615
ITHRESH = 0.06505
JTHRESH = 0.01030
KTHRESH = 0.01830
LTHRESH = 0.04020
MTHRESH = 0.02550
NTHRESH = 0.08355
OTHRESH = 0.06755
PTHRESH = 0.01575
QTHRESH = 0.00050
RTHRESH = 0.05655
STHRESH = 0.04985
TTHRESH = 0.07890
UTHRESH = 0.02485
VTHRESH = 0.01650
WTHRESH = 0.02030
XTHRESH = 0.00125
YTHRESH = 0.01050
ZTHRESH = 0.00830

# representative characters
SPACE = ' '
PIPE = '|'
EMPTY = '.'
ENCHAR = '+'
NLCHAR = '-'
ADADEL = ','

# command line inputs
func = ''
examples = ''
hypothesis = ''
learntype = ''
datafile = ''

'''
Writes constituent hypothesis values to file.
@param hypolist list of constituent hypothesis values
@param f        opened file to edit
'''
def adatofile(hypolist, f):
    for hypo in hypolist:
        f.write(ADADEL)
        f.write(str(hypo))

'''
Creates a list of constituent hypothesis values.
@param excount  number of examples
@param attrlist list of each examples' attribute tuples
@param langlist list of the languages for every example
'''
def adahypo(excount, attrlist, langlist):
    hypolist = []
    weightlist = []
    changelist = []
    wi = 1 / excount
    for i in range(excount):
        weightlist.append(wi)
        changelist.append(True)
    hcount = 0
    for a in range(26):
        if hcount >= HMAX:
            break
        aset = []
        bset = []
        for i in range(excount):
            if attrlist[i][a]:
                aset.append(i)
            else:
                bset.append(i)
        for i in [EN, NL]:
            for j in [EN, NL]:
                if hcount >= HMAX:
                    break
                err = 0
                for k in aset:
                    changelist[k] = (langlist[k] == i)
                    if changelist[k]:
                        err += weightlist[k]
                for k in bset:
                    changelist[k] = (langlist[k] == j)
                    if changelist[k]:
                        err += weightlist[k]
                err = min(max(.0000000000000001, err), .9999999999999999)
                update = err / (1 - err)
                if 0 < round(err, 3) < 1:
                    for k in range(excount):
                        if changelist[k]:
                            weightlist[k] *= update
                    s = sum(weightlist)
                    for k in range(len(weightlist)):
                        weightlist[k] /= s
                hypolist.append(math.log(1 / update))
                hcount += 1
    return hypolist

'''
Finds the index of the node's left child in an array representation.
@param n    index of the parent node
'''
def lnode(n):
    return 2 * n + 1

'''
Finds the index of the node's right child in an array representation.
@param n    index of the parent node
'''
def rnode(n):
    return 2 * n + 2

'''
Writes a representation of the decision tree to file.
@param node     the tuple representing the current node to write
@param f        the file to write into
'''
def treetofile(node, f):
    lang = node[0]
    if lang == EN:
        f.write(ENCHAR)
    elif lang == NL:
        f.write(NLCHAR)
    else:
        i = f.tell()
        f.write(node[1])
        f.seek(lnode(i))
        childa = node[2]
        childb = node[3]
        if childa != None:
            treetofile(childa, f)
            f.seek(rnode(i))
        else:
            f.write(EMPTY)
        if childb != None:
            treetofile(childb, f)
        else:
            f.write(EMPTY)

'''
Sets up the hypothesis file for filling.
@param f    the file to set up
'''
def resetfile(f):
    f.truncate(0)
    if learntype == DT:
        f.write(EMPTY * 2047)
        f.seek(0)

'''
Finds the entropy of a feature.
@param langlist     list of languages for every example
@param aset         list of examples where the feature is true
@param bset         list of examples where the feature is false
'''
def entropy(langlist, aset, bset):
    asize = len(aset)
    aen = 0
    anl = 0
    for i in aset:
        if langlist[i] == EN:
            aen += 1
        else:
            anl += 1
    a = 0
    if aen == 0:
        if anl != 0:
            a = anl * math.log2(asize / anl)
    elif anl == 0:
        a = aen * math.log2(asize / aen)
    else:
        a = aen * math.log2(asize / aen) + anl * math.log2(asize / anl)
    bsize = len(bset)
    ben = 0
    bnl = 0
    for i in bset:
        if langlist[i] == EN:
            ben += 1
        else:
            bnl += 1
    b = 0
    if ben == 0:
        if bnl != 0:
            b = bnl * math.log2(bsize / bnl)
    elif bnl == 0:
        b = ben * math.log2(bsize/ ben)
    else:
        b = ben * math.log2(bsize / ben) + bnl * math.log2(bsize / bnl)
    return (a + b) / (asize + bsize)

'''
Builds a decision tree.
@param unused       list of available features
@param exlist       list of examples in this node
@param attrlist     list of examples' attribute tuples
@param langlist     list of languages for every example
@param depth        remaining depth of the tree
'''
def buildtree(unused, exlist, attrlist, langlist, depth):
    nodelang = None
    split = None
    childa = None
    childb = None
    en = 0
    nl = 0
    for i in exlist:
        if langlist[i] == EN:
            en += 1
        else:
            nl += 1
    if len(unused) == 0 or depth == 0:
        if en < nl:
            nodelang = NL
        else:
            nodelang = EN
    elif en == 0:
        nodelang = NL
    elif nl == 0:
        nodelang = EN
    else:
        champ = unused[0]
        minent = 1
        champaset = []
        champbset = []
        for i in unused:
            aset = []
            bset = []
            for j in exlist:
                if attrlist[j][i]:
                    aset.append(j)
                else:
                    bset.append(j)
            ent = entropy(langlist, aset, bset)
            if ent < minent:
                champ = i
                minent = ent
                champaset = aset
                champbset = bset
        unused.remove(champ)
        split = CAPSALPHA[champ]
        depth -= 1
        if len(champaset) > 0:
            childa = buildtree(unused.copy(), champaset, attrlist, langlist, depth)
        if len(champbset) > 0:
            childb = buildtree(unused.copy(), champbset, attrlist, langlist, depth)
    node = (nodelang, split, childa, childb)
    return node

'''
Finds the attributes in all lines of text provided.
'''
def findattr(textlist):
    attrlist = []
    for text in textlist:
        text = text.upper()
        redtext = ''
        for c in text:
            if c.isalpha() or (c == SPACE and not redtext.endswith(SPACE)):
                redtext += c
            elif c in DUTCHEXTRA:
                if c == DUTCHEXTRA[0]:
                    redtext += 'A'
                elif c in (DUTCHEXTRA[1], DUTCHEXTRA[2], DUTCHEXTRA[3]):
                    redtext += 'E'
                elif c == DUTCHEXTRA[4]:
                    redtext += 'I'
                elif c == DUTCHEXTRA[5]:
                    redtext += 'IJ'
                elif c == DUTCHEXTRA[6]:
                    redtext += 'O'
                else:
                    redtext += 'U'
        length = len(redtext) - redtext.count(SPACE)
        aq = (redtext.count(CAPSALPHA[0]) / length) > ATHRESH
        bq = (redtext.count(CAPSALPHA[1]) / length) > BTHRESH
        cq = (redtext.count(CAPSALPHA[2]) / length) > CTHRESH
        dq = (redtext.count(CAPSALPHA[3]) / length) > DTHRESH
        eq = (redtext.count(CAPSALPHA[4]) / length) > ETHRESH
        fq = (redtext.count(CAPSALPHA[5]) / length) > FTHRESH
        gq = (redtext.count(CAPSALPHA[6]) / length) > GTHRESH
        hq = (redtext.count(CAPSALPHA[7]) / length) > HTHRESH
        iq = (redtext.count(CAPSALPHA[8]) / length) > ITHRESH
        jq = (redtext.count(CAPSALPHA[9]) / length) > JTHRESH
        kq = (redtext.count(CAPSALPHA[10]) / length) > KTHRESH
        lq = (redtext.count(CAPSALPHA[11]) / length) > LTHRESH
        mq = (redtext.count(CAPSALPHA[12]) / length) > MTHRESH
        nq = (redtext.count(CAPSALPHA[13]) / length) > NTHRESH
        oq = (redtext.count(CAPSALPHA[14]) / length) > OTHRESH
        pq = (redtext.count(CAPSALPHA[15]) / length) > PTHRESH
        qq = (redtext.count(CAPSALPHA[16]) / length) > QTHRESH
        rq = (redtext.count(CAPSALPHA[17]) / length) > RTHRESH
        sq = (redtext.count(CAPSALPHA[18]) / length) > STHRESH
        tq = (redtext.count(CAPSALPHA[19]) / length) > TTHRESH
        uq = (redtext.count(CAPSALPHA[20]) / length) > UTHRESH
        vq = (redtext.count(CAPSALPHA[21]) / length) > VTHRESH
        wq = (redtext.count(CAPSALPHA[22]) / length) > WTHRESH
        xq = (redtext.count(CAPSALPHA[23]) / length) > XTHRESH
        yq = (redtext.count(CAPSALPHA[24]) / length) > YTHRESH
        zq = (redtext.count(CAPSALPHA[25]) / length) > ZTHRESH
        attr = (aq, bq, cq, dq, eq, fq, gq, hq, iq, jq, kq, lq, mq, \
                nq, oq, pq, qq, rq, sq, tq, uq, vq, wq, xq, yq, zq)
        attrlist.append(attr)
    return attrlist

'''
Formulates a hypothesis with training data.
'''
def train():
    exlist = []
    with open(examples) as f:
        exlist = f.readlines()
    excount = len(exlist)
    langlist = []
    textlist = []
    for ex in exlist:
        lang, text = ex.split(PIPE, 1)
        langlist.append(lang)
        textlist.append(text)
    attrlist = findattr(textlist)
    if learntype == DT:
        root = buildtree(list(range(26)), list(range(excount)), attrlist, \
                langlist, MAXDEPTH)
        with open(hypothesis, 'w+') as f:
            resetfile(f)
            treetofile(root, f)
    else:
        hypolist = adahypo(excount, attrlist, langlist)
        with open(hypothesis, 'w+') as f:
            resetfile(f)
            adatofile(hypolist, f)

'''
Tells language predictions of requested text.
'''
def predict():
    content = ''
    with open(hypothesis, 'r') as f:
        content = f.read()
    textlist = ''
    with open(datafile, 'r') as f:
        textlist = f.readlines()
    attrlist = findattr(textlist)
    if content[0] != ADADEL:
        for attr in attrlist:
            found = False
            i = 0
            while not found:
                c = content[i]
                if c == ENCHAR:
                    print(EN)
                    found = True
                elif c == NLCHAR:
                    print(NL)
                    found = True
                elif attr[CAPSALPHA.find(c)]:
                    i = lnode(i)
                else:
                    i = rnode(i)
    else:
        hypolist = content.split(ADADEL)[1:]
        for i in range(len(hypolist)):
            hypolist[i] = float(hypolist[i])
        for attr in attrlist:
            calc = 0
            for i in range(len(attr)):
                j = 4 * i
                off = hypolist[j] + hypolist[j + 1] \
                        + hypolist[j + 2] + hypolist[j + 3]
                if attr[i]:
                    calc += off
                else:
                    calc -= off
            if calc > 0:
                print(EN)
            else:
                print(NL)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('ERROR: too few args')
    func += sys.argv[1]
    if func == 'train':
        if len(sys.argv) < 5:
            sys.exit('ERROR: too few args for train')
        examples = sys.argv[2]
        hypothesis = sys.argv[3]
        learntype = sys.argv[4]
        if learntype not in (DT, ADA):
            sys.exit('ERROR: learntype must be \'dt\' or \'ada\'')
        train()
    elif func == 'predict':
        if len(sys.argv) < 4:
            sys.exit('ERROR: too few args for predict')
        hypothesis = sys.argv[2]
        datafile = sys.argv[3]
        predict()
    else:
        sys.exit('ERROR: entry point may be \'train\' or \'predict\'')
