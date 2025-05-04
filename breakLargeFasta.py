import pickle
fileOpen = open('/data/data/fasta/fasta')


fastaDict={}
for f in fileOpen:
    if (f.startswith('>')):
        name=f
    else:
        fastaDict[name[1:].strip()]=f

print(fastaDict['A2RUR9'])

pickle.dump( fastaDict, open( "/data/data/fasta.p", "wb" ) )
