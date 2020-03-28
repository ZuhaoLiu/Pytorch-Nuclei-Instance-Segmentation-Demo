import numpy as np
import SimpleITK as sitk

def F1score(S, G):
	"""calculate instance F1 score
	S: instance prediction
	G: label
	"""
	S = S.astype(np.float32)
	G = G.astype(np.float32)
	
	listS = np.unique(S)
	listS = listS[listS != 0]
	numS = listS.shape[0]
	
	listG = np.unique(G)
	listG = listG[listG != 0]
	numG = listG.shape[0]

	if numS == 0 or numG == 0:
		return False
		
	tempMat = np.zeros([numS, 3])#the 1st col contains labels of segmented objects
				#the 2nd col contains labels of the corresponding ground truth objects
				#the 3rd col contains true positive flags
	tempMat[:, 0] = listS

	for iSegmentedObj in range(numS):
		intersectGTObjs = G[S == tempMat[iSegmentedObj, 0]]
		intersectGTObjs = intersectGTObjs[intersectGTObjs != 0]
		if intersectGTObjs.size != 0:

			tempMat[iSegmentedObj, 1] = np.argmax(np.bincount(intersectGTObjs.astype(np.int32))).astype(intersectGTObjs.dtype)
	
	for iSegmentedObj in range(numS):
		if tempMat[iSegmentedObj, 1] != 0:
			SegObj = S == tempMat[iSegmentedObj, 0]
			GTObj = G == tempMat[iSegmentedObj, 1]
			overlap = np.logical_and(SegObj, GTObj)
			areaOverlap = float(np.sum(overlap))
			areaGTObj = float(np.sum(GTObj))
			if areaOverlap / areaGTObj > 0.5:
				tempMat[iSegmentedObj, 2] = 1

	TP = float(np.sum(tempMat[:, 2]))
	FP = float(np.sum((tempMat[:, 2] == 0)))
	FN = numG - TP
	
	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	if precision == 0 and recall == 0:
		score = 0
	else:
		score = (2 * precision * recall) / (precision + recall)
	return [score, precision, recall]

def ObjectDice(S, G):
	"""calculate instance dice
	S: instance prediction
	G: label
	"""
	def Dice(A, B):
		inter = float(np.sum(np.logical_and(A, B)))
		union = float(np.sum(A)) + float(np.sum(B))
		return 2 * inter / union 
	S = S.astype(np.float32)
	G = G.astype(np.float32)
	
	listLabelS = np.unique(S)
	listLabelS = listLabelS[listLabelS != 0]
	numS = listLabelS.shape[0]

	listLabelG = np.unique(G)
	listLabelG = listLabelG[listLabelG != 0]
	numG = listLabelG.shape[0]
	
	if numS == 0 or numG == 0:
		return False

	temp1 = 0
	totalAreaS = float(np.sum(S>0))
	
	for iLabelS in range(numS):
		Si = S == listLabelS[iLabelS]
		intersectlist = G[Si]
		intersectlist = intersectlist[intersectlist != 0]

		if intersectlist.size != 0:
			indexGi = np.argmax(np.bincount(intersectlist.astype(np.int32))).astype(intersectlist.dtype)
			Gi = G == indexGi
		else:
			Gi = np.ones_like(G)
			Gi = Gi == 0

		omegai = float(np.sum(Si)) / totalAreaS
		temp1 += omegai*Dice(Gi, Si)

	temp2 = 0
	totalAreaG = float(np.sum(G>0))

	for iLabelG in range(numG):
		tildeGi = G == listLabelG[iLabelG]
		intersectlist = S[tildeGi]
		intersectlist = intersectlist[intersectlist != 0]

		if intersectlist.size != 0:
			indextildeSi = np.argmax(np.bincount(intersectlist.astype(np.int32))).astype(intersectlist.dtype)
			tildeSi = S == indextildeSi
		else:
			tildeSi = np.ones_like(S)
			tildeSi = tildeSi == 0

		tildeOmegai = float(np.sum(tildeGi)) / totalAreaG
		temp2 += tildeOmegai * Dice(tildeGi, tildeSi)

	objDice = (temp1 + temp2) / 2
	return objDice


def ObjectHausdorff(S, G):
	"""calculate instance Hausdorff distance
	S: instance prediction
	G: label
	"""
	def Hausdorff(A, B):
		A = A.astype(np.int32)
		B = B.astype(np.int32)
		img1 = sitk.GetImageFromArray(A)
		img2 = sitk.GetImageFromArray(B)
		hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
		hausdorffcomputer.Execute(img1, img2)
		return hausdorffcomputer.GetHausdorffDistance()

	S = S.astype(np.float32)
	G = G.astype(np.float32)

	tempS = S > 0
	totalAreaS = float(np.sum(tempS))

	tempG = G > 0
	totalAreaG = float(np.sum(tempG))

	listLabelS = np.unique(S)
	listLabelS = listLabelS[listLabelS != 0]

	listLabelG = np.unique(G)
	listLabelG = listLabelG[listLabelG != 0]
	
	# calculate object-level Hausdorff distance
	
	temp1 = 0

	for iLabelS in range(listLabelS.shape[0]):
		Si = S == listLabelS[iLabelS]
		intersectlist = G[Si]
		intersectlist = intersectlist[intersectlist != 0]

		if intersectlist.size != 0:
			indexGi = np.argmax(np.bincount(intersectlist.astype(np.int32))).astype(intersectlist.dtype)
			Gi = G == indexGi
		else:
			tempDist = np.zeros([listLabelG.shape[0], 1])
			for iLabelG in range(listLabelG.shape[0]):
				Gi = G == listLabelG[iLabelG]
				tempDist[iLabelG] = Hausdorff(Gi, Si)
			minIdx = np.argmin(tempDist)
			Gi = G == listLabelG[minIdx]
		omegai = float(np.sum(Si)) / totalAreaS
		temp1 += omegai * Hausdorff(Gi, Si)

	temp2 = 0

	for iLabelG in range(listLabelG.shape[0]):
		tildeGi = G == listLabelG[iLabelG]
		intersectlist = S[tildeGi]
		intersectlist = intersectlist[intersectlist != 0]

		if intersectlist.size != 0:
			indextildeSi = np.argmax(np.bincount(intersectlist.astype(np.int32))).astype(intersectlist.dtype)
			tildeSi = S == indextildeSi
		else:
			tempDist = np.zeros([listLabelS.shape[0], 1])
			for iLabelS in range(listLabelS.shape[0]):
				tildeSi = S == listLabelS[iLabelS]
				tempDist[iLabelS] = Hausdorff(tildeGi, tildeSi)
			minIdx = np.argmin(tempDist)
			tildeSi = S == listLabelS[minIdx]

		tildeOmegai = float(np.sum(tildeGi)) / totalAreaG
		temp2 += tildeOmegai * Hausdorff(tildeGi, tildeSi)
	objHausdorff = (temp1 +temp2) / 2
	return objHausdorff
































