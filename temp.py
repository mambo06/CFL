import numpy as np

alist = [
    [None, (0.5158370499147619, 0.5282588584582948, 0.4712824378761526, None)],
    [None, (0.5760299599284222, 0.5548011937289277, 0.5036150027377568, None)],
    [None, (0.5353380023918832, 0.5672025743237636, 0.5310297130847121, None)],
    [None, (0.690271248622301, 0.7023023232226993, 0.6918026326520008, None)]
]

oriPrecL, oriRecL, oriFL = [], [], []
embedPrecL, embedRecL, embedFL = [], [], []

for item in alist:
    ori, embed = item
    if ori is not None:
        oriPrec, oriRec, oriF, _ = ori
        oriPrecL.append(oriPrec)
        oriRecL.append(oriRec)
        oriFL.append(oriF)
    
    if embed is not None:
        embedPrec, embedRec, embedF, _ = embed
        embedPrecL.append(embedPrec)
        embedRecL.append(embedRec)
        embedFL.append(embedF)

# Calculate mean for embed results
if embedPrecL and embedRecL and embedFL:
    embed_mean = (
        f'Mean of embed results. \
        Precision : {np.mean(np.array(embedPrecL))} \
        Recall : {np.mean(np.array(embedRecL))} \
        F1 : {np.mean(np.array(embedFL))} '
    )
else:
    embed_mean = "No valid embed results to calculate the mean."

# Calculate mean for original results
if oriPrecL and oriRecL and oriFL:
    ori_mean = (
        f'Mean of original results. \
        Precision : {np.mean(np.array(oriPrecL))} \
        Recall : {np.mean(np.array(oriRecL))} \
        F1 : {np.mean(np.array(oriFL))} '
    )
else:
    ori_mean = "No valid original results to calculate the mean."

# Print the results
print(embed_mean)
print(ori_mean)
