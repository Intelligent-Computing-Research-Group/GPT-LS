
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
import numpy as np
import time, re
import networkx as nx
import torch_geometric
import torch_geometric.data


def processANDAssignments(inputs,output,idxCounter,poList,nodeNameIDMapping,singleGateInputIOMapping,AIG_DAG):
    nodeType = {
        "PI": 0,
        "PO": 1,
        "Internal": 2
    }

    edgeType = {
        "BUFF": 0,
        "NOT": 1,
    }
    nType = nodeType["Internal"]
    nodeAttributedDict = {
        "node_id": output,
        "node_type": nType,
        "num_inverted_predecessors": 0
    }
    AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
    nodeNameIDMapping[output] = idxCounter
    numInvertedPredecessors = 0
    for inp in inputs:
        if not (inp in nodeNameIDMapping.keys()):
            srcIdx = nodeNameIDMapping[singleGateInputIOMapping[inp]]
            eType = edgeType["NOT"]
            numInvertedPredecessors+=1
        else:
            srcIdx = nodeNameIDMapping[inp]
            eType = edgeType["BUFF"]
        AIG_DAG.add_edge(idxCounter,srcIdx,edge_type=eType)
    AIG_DAG.nodes[idxCounter]["num_inverted_predecessors"] = numInvertedPredecessors

    # If output is primary output, add additional node to keep it consistent with POs having inverters
    if (output in poList):
        nType = nodeType["PO"]
        nodeAttributedDict = {
            "node_id": output+"_buff",
            "node_type": nType,
            "num_inverted_predecessors": 0
        }
        AIG_DAG.add_nodes_from([(idxCounter+1, nodeAttributedDict)])
        nodeNameIDMapping[output+"_buff"] = idxCounter+1
        srcIdx = idxCounter
        eType = edgeType["BUFF"]
        AIG_DAG.add_edge(idxCounter+1, srcIdx, edge_type=eType)

def parseAIGBenchAndCreateNetworkXGraph(INPUT_BENCH):
    nodeType = {
        "PI": 0,
        "PO": 1,
        "Internal": 2
    }

    edgeType = {
        "BUFF": 0,
        "NOT": 1,
    }
    nodeNameIDMapping = {}
    singleInputgateIOMapping = {}
    poList = []
    benchFile = open(INPUT_BENCH,'r+')
    benchFileLines = benchFile.readlines()
    benchFile.close()
    AIG_DAG = nx.DiGraph()
    idxCounter = 0
    for line in benchFileLines:
        if len(line) == 0 or line.__contains__("ABC"):
            continue
        elif line.__contains__("vdd"):
            ## Treat Vdd assignment as Primary Input.
            line = line.replace(" ","")
            pi = re.search("(.*?)=", str(line)).group(1)
            nodeAttributedDict = {
                "node_id": pi,
                "node_type": nodeType["PI"],
                "num_inverted_predecessors": 0
            }
            AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
            nodeNameIDMapping[pi] = idxCounter
            idxCounter+=1
        elif line.__contains__("INPUT"):
            line = line.replace(" ","")
            pi = re.search("INPUT\((.*?)\)",str(line)).group(1)
            nodeAttributedDict = {
                "node_id": pi,
                "node_type": nodeType["PI"],
                "num_inverted_predecessors": 0
            }
            AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
            nodeNameIDMapping[pi] = idxCounter
            idxCounter+=1
        elif line.__contains__("OUTPUT"):
            line = line.replace(" ", "")
            po = re.search("OUTPUT\((.*?)\)", str(line)).group(1)
            poList.append(po)
        elif line.__contains__("AND"):
            line = line.replace(" ", "")
            output = re.search("(.*?)=", str(line)).group(1)
            input1 = re.search("AND\((.*?),",str(line)).group(1)
            input2 = re.search(",(.*?)\)", str(line)).group(1)
            processANDAssignments([input1,input2], output, idxCounter, poList, nodeNameIDMapping, singleInputgateIOMapping, AIG_DAG)
            if output in poList:
                idxCounter += 1
            idxCounter+=1
        elif line.__contains__("NOT"):
            line = line.replace(" ", "")
            output = re.search("(.*?)=", str(line)).group(1)
            inputPin = re.search("NOT\((.*?)\)", str(line)).group(1)
            singleInputgateIOMapping[output] = inputPin
            if output in poList:
                nodeAttributedDict = {
                    "node_id": output+"_inv",
                    "node_type": nodeType["PO"],
                    "num_inverted_predecessors": 1
                }
                AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
                nodeNameIDMapping[output+"_inv"] = idxCounter
                srcIdx = nodeNameIDMapping[inputPin]
                eType = edgeType["NOT"]
                AIG_DAG.add_edge(idxCounter, srcIdx, edge_type=eType)
                idxCounter += 1
        elif line.__contains__("BUFF"):
            line = line.replace(" ", "")
            output = re.search("(.*?)=", str(line)).group(1)
            inputPin = re.search("BUFF\((.*?)\)", str(line)).group(1)
            singleInputgateIOMapping[output] = inputPin
            numInvertedPredecessors = 0
            if output in poList:
                ## Additional logic: Input pin may be inverter. So perform an initial check whether inputPin in available
                ## in nodeNameIDMapping or not
                if inputPin in nodeNameIDMapping.keys():
                    srcIdx = nodeNameIDMapping[inputPin]
                    eType = edgeType["BUFF"]
                else:
                    ## instance of NOT gate followed by BUFF gates.
                    srcIdx = nodeNameIDMapping[singleInputgateIOMapping[inputPin]]
                    eType = edgeType["NOT"]
                    numInvertedPredecessors+=1
                nodeAttributedDict = {
                    "node_id": output+"_buff",
                    "node_type": nodeType["PO"],
                    "num_inverted_predecessors": numInvertedPredecessors
                }
                AIG_DAG.add_nodes_from([(idxCounter, nodeAttributedDict)])
                nodeNameIDMapping[output+"_buff"] = idxCounter
                AIG_DAG.add_edge(idxCounter, srcIdx, edge_type=eType)
                idxCounter += 1
        else:
            print(" Line contains unknown characters.", line)
            exit(1)
    return AIG_DAG
def pygDataFromNetworkx(G):
    nodeType = {
        0: "PI",
        1: "PO",
        2: "Internal"
    }
    edgeType = {1: "NOT", 0: 'BUFF'}
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.
    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = {}
    nodeCountDict = {"PI": 0, "PO": 0, "Internal": 0}
    edgeCountDict = {'BUFF': 0, 'NOT': 0}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]
        nodeCountDict[nodeType[feat_dict['node_type']]] += 1

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]
        edgeCountDict[edgeType[feat_dict['edge_type']]] += 1

    data['longest_path'] = nx.dag_longest_path_length(G)
    data['and_nodes'] = nodeCountDict["Internal"]
    data['pi'] = nodeCountDict["PI"]
    data['po'] = nodeCountDict["PO"]
    data['not_edges'] = edgeCountDict["NOT"]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data
def getMeanAndVariance(targetList):
    return np.mean(np.array(targetList)),np.std(np.array(targetList))

def computeMeanAndVarianceOfTargets(targetStatsDict,targetVar='nodes'):
    meanAndVarTargetDict = {}
    for des in targetStatsDict.keys():
        numNodes,NotGates,depth,areaVar,delayVar = targetStatsDict[des]
        #print(targetStatsDict[des])
        if targetVar == 'delay':
            meanTarget,varTarget = getMeanAndVariance(delayVar)
        elif targetVar == 'area':
            meanTarget,varTarget = getMeanAndVariance(areaVar)
        else:
            meanTarget,varTarget = getMeanAndVariance(numNodes)
            meanDepth,varDepth = getMeanAndVariance(depth)
        meanAndVarTargetDict[des] = [meanTarget,varTarget, meanDepth, varDepth]
    return meanAndVarTargetDict

def addNormalizedTargets(data,targetStatsDict,meanVarDataDict,targetVar='nodes'):
    sid = data.synID[0]
    #print('data:', data)
    #print('targetStatsDict:', targetStatsDict)
    #print('sid:', sid)
    desName = data.desName[0]
    #print('desName:', desName)  # mem_ctrl
    if targetVar == 'delay':    
        targetIdentifier = 4 # Column number of target 'Delay' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    elif targetVar == 'area':
        targetIdentifier = 3 # Column number of target 'Area' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    else:
        targetIdentifier = 0 # Column number of target 'Nodes' in synthesisStatistics.pickle entries
        #print(targetStatsDict[desName][targetIdentifier][sid])

        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]

        data.target = torch.tensor([normTarget], dtype=torch.float32)
    return data

def addNormalizedStates(data, targetStatsDict, meanVarDataDict, targetVar='nodes'):
    sid = data.synID[0]
    #print('data:', data)
    #print('targetStatsDict:', targetStatsDict)
    #print('sid:', sid)
    desName = data.desName[0]
    #print('desName:', desName)  # mem_ctrl
    if targetVar == 'delay':
        targetIdentifier = 4 # Column number of target 'Delay' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    elif targetVar == 'area':
        targetIdentifier = 3 # Column number of target 'Area' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    else:
        targetIdentifier = 0 # Column number of target 'Nodes' in synthesisStatistics.pickle entries
        #print(targetStatsDict[desName][targetIdentifier][sid])

        #normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        #normANDgates = (targetStatsDict[desName][0][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        #normDepth = (targetStatsDict[desName][2][sid] - meanVarDataDict[desName][2]) / meanVarDataDict[desName][3]
        #data = data[:2] + data[3:]
        #data.target = torch.tensor([normTarget], dtype=torch.float32)
        #data.normANDgates = torch.tensor([normANDgates], dtype=torch.float32)
        #data.normDepth = torch.tensor([normDepth], dtype=torch.float32)

        start_idx = data.synID[0]*21
        #start_idx = count*21
        #print(DS[0]['stepID'])

        #print(torch.tensor([data.and_nodes]))
        #print(torch.tensor([DS[0]['and_nodes']]))
        # data.states = torch.tensor([data.and_nodes])
        #data.states = torch.tensor([data.normANDgates, data.normDepth], dtype=torch.float32)
        #data.rtg = torch.tensor([[data.and_nodes]], dtype=torch.float32)

        data.dones = torch.zeros(21)
        data.dones[-1] = 1
        data.timestep = torch.tensor([[0]])


        # data.edge_indexs = data.edge_index
        # data.node_types = data.node_type
        # data.num_inverted_predecessorses = data.num_inverted_predecessors

        #single_dl = DataLoader(DS, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)
        #print('single_dl:', single_dl)
        # for _, batch in enumerate(single_dl):
        #     print(batch)
        #print(single_dl[0])

        rtg_list = []
        graph_state_list = []
        rtg_file = '/home/lcy/PycharmProjects/OPENABC2_DATASET/rtg_data/' + desName + '_full_len20.pth'
        rtg_list = torch.load(rtg_file)
        graph_file = '/home/lcy/PycharmProjects/OpenABC/models/qor/SynthNetV5-GCN+DecisionTransformer/DecisionTransformer/Graph_extracted_data/' + desName + '_full_state128.pth'
        graph_state_list = torch.load(graph_file)

        #print(rtg_list)
        data.rtg_list = torch.tensor(rtg_list[start_idx:start_idx + 20]).squeeze()
        #data.rtg_list = torch.stack(rtg_list[start_idx:start_idx + 20]).squeeze()
        graph_state = torch.stack(graph_state_list[start_idx:start_idx + 20]).squeeze()
        data.graph_state = torch.tensor(graph_state, dtype=torch.float32)
        #data.graph_state_pred = torch.stack(graph_state_list[start_idx+1:start_idx+21]).squeeze()
        #print(data)
        data.edge_index = None
        data.edge_type = None
        data.node_id = None
        data.node_type = None
        data.num_inverted_predecessors = None
        #print(data.desName)
        #print(data)
        #input()
        #data.graph_state =
        #data.graph_state = graph_model(single_dl.__iter__())
        #print(data.graph_state)
        #print('ooook')

        for i in range(19):
            # print('synID:', DS[int(start_idx + i)]['synID'][0])
            # print('stepID:', DS[int(start_idx + i)]['stepID'][0])


            # normed_and = (DS[int(start_idx + i)]['and_nodes'] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
            # normed_depth = (DS[int(start_idx + i)]['longest_path'] - meanVarDataDict[desName][2]) / meanVarDataDict[desName][3]
            # next_state = torch.tensor([normed_and, normed_depth])
            #DS[int(start_idx + i)]['longest_path']
            # next_rtg = torch.tensor([[DS[int(start_idx+i)]['and_nodes']]], dtype=torch.float32)
            #print('next_state_vector:', next_state)
            # data.states = torch.cat([data.states, next_state], 0)
            # data.rtg = torch.cat((data.rtg, next_rtg), -1)
            #print(data.rtg)
            data.timestep = torch.cat((data.timestep, torch.tensor([[i]])), -1)
            #print(data.stepID)
            #data.stepID = torch.cat((data.stepID, DS[int(start_idx + i)]['stepID'][0]), -1)
            # data.edge_indexs = torch.cat((data.edge_indexs, DS[int(start_idx + i)].edge_index), -1)
            # data.node_types = torch.cat((data.node_types, DS[int(start_idx + i)].node_type), -1)
            # data.num_inverted_predecessorses = torch.cat((data.num_inverted_predecessorses, DS[int(start_idx + i)].num_inverted_predecessors), -1)


            # print(DS[int(start_idx + i)])
            # print(data)

        #data.states = torch.cat((DS[0]['and_nodes'], DS[1]['and_nodes']), 0)
        #data.target.append(torch.tensor([normTarget],dtype=torch.float32))
        # print(data)
        # print(data['rtg_list'])
        # print(data['desName'])
        # input()
    return data

def addAbsoluteTargets(data, targetStatsDict, targetVar='nodes'):
    sid = data.synID[0]
    desName = data.desName[0]
    numNodes,_,_,areaVar,delayVar = targetStatsDict[desName]
    if targetVar == 'delay':
        data.target = torch.tensor([delayVar[sid]],dtype=torch.float32)
    elif targetVar == 'area':
        data.target = torch.tensor([areaVar[sid]],dtype=torch.float32)
    else:
        data.target = torch.tensor([numNodes[sid]],dtype=torch.float32)
    return data

# Torch.std_mean returns tuple with std first and mean second term
def mapMeanChangeToTensor(data,areaStatsDict,delayStatsDict):
    area = data.area
    delay = data.delay
    data.area = (area - areaStatsDict[data.desName[0]][1]) / areaStatsDict[data.desName[0]][0]
    data.delay = (delay - delayStatsDict[data.desName[0]][1]) / delayStatsDict[data.desName[0]][0]
    assert(data.area > -10 and data.area < 10)
    return data

# Element 0 is area and 1 is delay
def getMeanAreaAndDelay(trainDS,testDS):
    desNamesTrain = set(elem.desName[0] for elem in trainDS)
    desNamesTest = set(elem.desName[0] for elem in testDS)
    desNameTotal = desNamesTrain.union(desNamesTest)
    desStatsArea = {}
    desStatsDelay = {}
    delayStats = {}
    areaStats = {}
    for des in desNameTotal:
        desStatsArea[des] = []
        desStatsDelay[des] = []
    for elem in trainDS:
        desStatsArea[elem.desName[0]].append(elem.area)
        desStatsDelay[elem.desName[0]].append(elem.delay)
    for elem in testDS:
        desStatsArea[elem.desName[0]].append(elem.area)
        desStatsDelay[elem.desName[0]].append(elem.delay)
    for des in desNameTotal:
        areaStats[des] = torch.std_mean(torch.tensor(desStatsArea[des]))
        delayStats[des] = torch.std_mean(torch.tensor(desStatsDelay[des]))
    return areaStats,delayStats


def getMinMaxTargetVal(dataSet):
    desMinMaxAreaVal = {}
    desMinMaxDelayVal = {}
    desNames = [elem.desName[0] for elem in dataSet]
    for des in desNames:
        desMinMaxAreaVal[des] = [None,None]
        desMinMaxDelayVal[des] = [None,None]
    for ditem in dataSet[1:]:
        des = ditem.desName[0]
        area = ditem.area
        delay = ditem.delay
        # Area computation
        desMinMaxAreaVal[des][0] = area if (area > desMinMaxAreaVal[des][0] or desMinMaxAreaVal[des][0] == None) else desMinMaxAreaVal[des][0]
        desMinMaxAreaVal[des][1] = area if (area < desMinMaxAreaVal[des][1] or desMinMaxAreaVal[des][1] == None) else desMinMaxAreaVal[des][1]
        # Delay computation
        desMinMaxDelayVal[des][0] = delay if (delay > desMinMaxDelayVal[des][0] or desMinMaxDelayVal[des][1] == None) else desMinMaxDelayVal[des][0]
        desMinMaxDelayVal[des][1] = delay if (delay < desMinMaxDelayVal[des][1] or desMinMaxDelayVal[des][1] == None) else desMinMaxDelayVal[des][1]
    return desMinMaxAreaVal,desMinMaxDelayVal

def checkUnseenDesInTest(areaDict,testDS):
    unseenDesigns = set(elem.desName[0] for elem in testDS if not elem.desName[0] in areaDict.keys())
    if len(unseenDesigns) > 0:
        desMinMaxAreaVal = {}
        desMinMaxDelayVal = {}
        for des in unseenDesigns:
            desMinMaxAreaVal[des] = [0, -1]
            desMinMaxDelayVal[des] = [0,-1]
        for ditem in testDS:
            des = ditem.desName[0]
            area = ditem.area
            delay = ditem.delay
            if( not des in unseenDesigns):
                pass
            # Area computation
            desMinMaxAreaVal[des][0] = area if area > desMinMaxAreaVal[des][0] else desMinMaxAreaVal[des][0]
            desMinMaxAreaVal[des][1] = area if (area < desMinMaxAreaVal[des][1] or area == -1) else desMinMaxAreaVal[des][1]
            # Delay computation
            desMinMaxDelayVal[des][0] = delay if delay > desMinMaxDelayVal[des][0] else desMinMaxDelayVal[des][0]
            desMinMaxDelayVal[des][1] = delay if (delay < desMinMaxDelayVal[des][1] or delay == -1) else desMinMaxDelayVal[des][1]
        return desMinMaxAreaVal, desMinMaxDelayVal
    else:
        return None,None


def getDevice():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def desName_to_idx(aigData):
    desNames = [elem.desName[0] for elem in aigData]
    desNameIdxDict = {}
    idxDesNameDict = {}
    i=0
    for des in desNames:
        if not des in desNameIdxDict.keys():
            desNameIdxDict[des] = i
            idxDesNameDict[i] = des
            i+=1
    return desNameIdxDict,idxDesNameDict

def mapNameToLabel(data,desNameIdxDict):
    labelName = data.desName[0]
    data.desLabel = torch.tensor([desNameIdxDict[labelName]])
    return data

def mapAttributesToTensor(data,areaDict,delayDict):
    area = data.area
    delay = data.delay
    minMaxArea = areaDict[data.desName[0]]
    minMaxDelay = delayDict[data.desName[0]]
    data.area = (area - minMaxArea[1])/(minMaxArea[0] - minMaxArea[1])
    data.delay = (delay - minMaxDelay[1]) / (minMaxDelay[0] - minMaxDelay[1])
    return data


def mse(y_pred,y_true):
    return mean_squared_error(y_true.view(-1,1).detach().cpu().numpy(),y_pred.view(-1,1).detach().cpu().numpy())

def mae(y_pred,y_true):
    return mean_absolute_error(y_true.view(-1,1).detach().cpu().numpy(),y_pred.view(-1,1).detach().cpu().numpy())

def doScatterPlot(batchLen,batchSize,batchData,dumpDir,trainMode):
    predList = []
    actualList = []
    designList = []
    for i in range(batchLen):
        numElemsInBatch = len(batchData[i][0])
        for batchID in range(numElemsInBatch):
            predList.append(batchData[i][0][batchID][0])
            actualList.append(batchData[i][1][batchID][0])
            designList.append(batchData[i][2][batchID][0])

    scatterPlotDF = pd.DataFrame({'designs': designList,
                                  'prediction': predList,
                                  'actual': actualList})

    uniqueDesignList = scatterPlotDF.designs.unique()

    for d in uniqueDesignList:
        designDF = scatterPlotDF[scatterPlotDF.designs == d]
        designDF.plot.scatter(x='actual', y='prediction', c='DarkBlue')
        plt.title(d)
        time_str = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
        fileName = osp.join(dumpDir, str(time_str) + "+scatterPlot_"+trainMode+"_"+d+".png")
        #else:
        #    fileName = osp.join(dumpDir,"scatterPlot_test_"+d+".png")
        plt.savefig(fileName,bbox_inches='tight')


def getTopKSimilarityPercentage(list1,list2,topkpercent):
    listLen = len(list1)
    topKIndexSimilarity = int(topkpercent*listLen)
    Set1 = set(list1[:topKIndexSimilarity])
    Set2 = set(list2[:topKIndexSimilarity])
    numSimilarScripts = len(Set1.intersection(Set2))
    if topKIndexSimilarity >0:
        return (numSimilarScripts/topKIndexSimilarity)
    else:
        return 0


def doScatterAndTopKRanking(batchLen,batchSize,batchData,dumpDir,trainMode, MSE):
    predList = []
    actualList = []
    designList = []
    synthesisID = []
    predList_ori = []
    actualList_ori = []
    for i in range(batchLen):
        numElemsInBatch = len(batchData[i][0])
        for batchID in range(numElemsInBatch):
            predList.append(batchData[i][0][batchID][0])
            actualList.append(batchData[i][1][batchID][0])
            designList.append(batchData[i][2][batchID][0])
            synthesisID.append(batchData[i][3][batchID][0])
            predList_ori.append(batchData[i][4][batchID][0])
            actualList_ori.append(batchData[i][5][batchID][0])

    scatterPlotDF = pd.DataFrame({'designs': designList,
                                  'synID': synthesisID,
                                  'prediction': predList,
                                  'actual': actualList,
                                  'prediction_ori': predList_ori,
                                  'actual_ori': actualList_ori
                                  })

    uniqueDesignList = scatterPlotDF.designs.unique()
    time_str = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
    accuracyFile = osp.join(dumpDir, str(time_str) + "topKaccuracy_" + trainMode + ".csv")
    accuracyFileWriter = open(accuracyFile, 'w+')
    accuracyFileWriter.write("design,top1,top5,top10,top15,top20,top25"+"\n")
    endDelim = "\n"
    commaDelim = ","

    print("\nDataset type: "+trainMode)
    for d in uniqueDesignList:
        designDF = scatterPlotDF[scatterPlotDF.designs == d]
        #designDF_ori = scatterPlotDF[scatterPlotDF.designs == d]
        designDF.plot.scatter(x='actual', y='prediction', c='DarkBlue')
        #designDF_ori.plot.scatter(x='actual_ori', y='prediction_ori', c='DarkBlue')
        plt.title(d+' '+ 'MSE:'+str("%.3f"%(MSE)), weight='bold', fontsize=25)
        plt.xlabel('Actual', weight='bold', fontsize=25)
        plt.ylabel('Predicted', weight='bold', fontsize=25)
        time_str = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
        fileName = osp.join(dumpDir, str(time_str) + "-scatterPlot_"+trainMode+"_"+d+".png")
        plt.savefig(fileName,bbox_inches='tight')
        desDF1 = designDF.sort_values(by=['actual'])
        desDF2 = designDF.sort_values(by=['prediction'])
        desDF1_synID = desDF1.synID.to_list()
        desDF2_synID = desDF2.synID.to_list()
        kPercentSimilarity = [0.01,0.05,0.1,0.15,0.2,0.25]
        accuracyFileWriter.write(d)
        for kPer in kPercentSimilarity:
            topKPercentSimilarity = getTopKSimilarityPercentage(desDF1_synID,desDF2_synID,kPer)
            accuracyFileWriter.write(commaDelim+str(topKPercentSimilarity))
        accuracyFileWriter.write(endDelim)
        time_str = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
        desDF1.to_csv(osp.join(dumpDir, str(time_str) + "-desDF1_"+trainMode+"_"+d+".csv"),index=False)
        desDF2.to_csv(osp.join(dumpDir, str(time_str) + "-desDF2_"+trainMode+"_"+d+".csv"),index=False)
        mapeScore = mean_absolute_percentage_error(designDF.prediction.to_list(), designDF.actual.to_list())
        print("MAPE ("+d+"): "+str(mapeScore))

    plt.clf()
    for d in uniqueDesignList:
        #designDF = scatterPlotDF[scatterPlotDF.designs == d]
        designDF_ori = scatterPlotDF[scatterPlotDF.designs == d]
        #designDF.plot.scatter(x='actual', y='prediction', c='DarkBlue')
        designDF_ori.plot.scatter(x='actual_ori', y='prediction_ori', c='DarkBlue')
        plt.title(d+' ' + 'MSE:'+str("%.3f"%(MSE)), weight='bold', fontsize=25)
        plt.xlabel('Actual', weight='bold', fontsize=25)
        plt.ylabel('Predicted', weight='bold', fontsize=25)
        time_str = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
        #fileName = osp.join(dumpDir, str(time_str) + "-scatterPlot_"+trainMode+"_"+d+".png")
        fileName_ori = osp.join(dumpDir, str(time_str) + "-scatterPlot_"+trainMode+"_"+d+"-ori.png")
        #plt.savefig(fileName,bbox_inches='tight')
        plt.savefig(fileName_ori,bbox_inches='tight')


    accuracyFileWriter.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count