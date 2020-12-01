import cv2
def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None, bb=True):
    tl = int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        # s_size = cv2.getTextSize(str('{:.0%}'.format(score)),0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=2*float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0]+15, c1[1] - t_size[1] -3
        if bb==True:
            cv2.rectangle(img, c1, c2 , color, -1)  # filled
        cv2.putText(img, '{}'.format(label), (c1[0],c1[1] - 2), 0, 2*float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

