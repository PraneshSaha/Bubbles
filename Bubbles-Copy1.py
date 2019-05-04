#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import numpy as np
import math


# In[9681]:


vidcap = cv2.VideoCapture("C:/Users/Pranesh/Desktop/vids/C001H001S0001/C001H001S0001.avi")
success,image = vidcap.read()
count = 0
success = True
vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
success,back = vidcap.read()
count += 1


# In[10793]:


vidcap = cv2.VideoCapture("C:/Users/Pranesh/Desktop/vids/C001H001S0009/C001H001S0009.avi")
success,image = vidcap.read()
count = 0
pos0=[]
pos1=[]
pos2=[]
success = True
while success:
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*20))
    success,img = vidcap.read()
    #mn0=np.mean(img[:50,:])
    #mn1=np.mean(img[375:425,:])
    #mn2=np.mean(img[800:850,:])
    #pos0.append(mn0)
    #pos1.append(mn1)
    #pos2.append(mn2)
    #print ('Read a new frame: ', success)
    cv2.imwrite( "C:/Users/Pranesh/Desktop/vid9/bubbles"+str(count)+".png", img)     # save frame as JPEG file
    count += 1
    if count%500==0:
        print(count)


# In[53]:


x_l=[]
for i in range(count):
    x_l.append(i)


# In[64]:


for i in range(0,count,550):
    #plt.plot(x_l[i:i+250],pos2[i:i+250],'r')
    plt.plot(x_l[i:i+550],pos1[i:i+550],'b')
    plt.plot(x_l[i:i+550],pos0[i:i+550],'g')
    plt.axis([i,i+550,25,85])
    plt.xlabel('Mean')
    plt.ylabel('Frame')

    plt.savefig('C:/Users/Pranesh/Desktop/Graphs/7pos'+str(i)+'.png')


# In[57]:


i


# In[11044]:


count=14000


# In[4437]:


all_frames.shape


# In[5]:


img0=cv2.imread( "C:/Users/Pranesh/Desktop/vid6/bubble0.png",0)


# In[10]:


from matplotlib.pyplot import imshow


# In[11133]:


imshow(im1)


# In[11045]:


def frame_catcher(vid_no,count):

    all_frames=np.zeros((img0.shape[0],img0.shape[1],(count)//50))
    for i in range(0,count,50):
        all_frames[:,:,(i)//50]=cv2.imread( "C:/Users/Pranesh/Desktop/vid"+str(vid_no)+"/bubbles"+str(i)+".png",0)
    for i in range(all_frames.shape[2]):
        t=all_frames[:,:,i].astype('uint8')
        all_frames[:,:,i]=cv2.bilateralFilter(t,9,75,75)
    
    return all_frames


# In[26]:


def segment(all_frames,count):
    n_img=np.ones((896,896))*255
    

    im1=all_frames[:,:,0]
    for j in range(im1.shape[0]):
        for k in range(im1.shape[1]):
            if n_img[j,k]!=0:
                for l in range(1,all_frames.shape[2]):
                    if abs(all_frames[j,k,l]-im1[j,k])==0:
                        n_img[j,k]=0
    kernel=np.ones((3,3))
    d = cv2.dilate(n_img,kernel,iterations = 2)
    new_img=d
    kernel=np.ones((3,3))
    e=cv2.erode(new_img,kernel,iterations=1)
    new_img2=e
    v=new_img2.astype('uint8')
    v=cv2.medianBlur(v,11)
    v=cv2.medianBlur(v,9)
    v=cv2.medianBlur(v,5)
    t=cv2.Canny(v,0,255)
    v=t
    x_b=[]
    y_b=[]
    for i in range(1,v.shape[0]-1):
        for j in range(1,v.shape[1]-1):
            if v[i,j]==255:
                x_b.append(i)
                y_b.append(j)
    
    #cv2.imwrite( "C:/Users/Pranesh/Desktop/segmented"+str(count)+".png", v)
    return x_b,y_b,v


# In[12]:


def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9
def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)


# In[13]:


def Thinned(v,x_b,y_b,count):
    Image_Thinned = v.copy()//255  # deepcopy to protect the original image
    changing1 = changing2 = 1   
    ind=[1]*len(x_b)
    #  the points to be removed (set as 0)
    while changing1!=0 or changing2!=0:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
    
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for i in range(len(x_b)):                     # No. of  rows
            x=x_b[i]
            y=y_b[i]
            P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
            if (Image_Thinned[x][y] == 1     and 
                ind[i]==1 and   # Condition 0: Point P1 in the object regions 
                2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                transitions(n) == 1 and    # Condition 2: S(P1)=1  
                P2 * P4 * P6 == 0  and    # Condition 3   
                P4 * P6 * P8 == 0):         # Condition 4
                changing1.append((x,y))
                ind[i]=0
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
    
    # Step 2
        changing2 = []
        for i in range(len(x_b)):
            x=x_b[i]
            y=y_b[i]
            P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
            if (Image_Thinned[x][y] == 1   and 
                ind[i]==1 and              # Condition 0
                2 <= sum(n) <= 6  and       # Condition 1
                transitions(n) == 1 and      # Condition 2
                P2 * P4 * P8 == 0 and       # Condition 3
                P2 * P6 * P8 == 0):            # Condition 4
                changing2.append((x,y))
                ind[i]=0
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
        changing1=len(changing1)
        changing2=len(changing2)
        
    new_img=Image_Thinned*255
    #if count%1==0:
        #cv2.imwrite( "C:/Users/Pranesh/Desktop/steps/Thinned"+str(count)+".png", new_img)
    return new_img


# In[20]:


def ending(new_img):
    new_img[0,:]=0
    new_img[:,0]=0
    new_img[new_img.shape[0]-1,:]=0
    new_img[:,new_img.shape[1]-1]=0
    x_end=[]
    y_end=[]
    count=0
    for i in range(1,new_img.shape[0]-1):
        for j in range(1,new_img.shape[1]-1):
             if new_img[i,j]==255:
                    kernel=new_img[i-1:i+2,j-1:j+2]
                    s=np.sum(kernel)//255
                #print(s)
                    if s==1:
                        new_img[i,j]=0
                    if s==2:
                        x_end.append(i)
                        y_end.append(j)
                        count=0
                    count+=1
                    
                    if count>10:
                        x_end.append(i)
                        y_end.append(j)
                        count=0
                        
    return x_end,y_end,new_img


# In[ ]:





# In[ ]:





# In[15]:


def branches(kernel,x,y):
    
    x_un,y_un=np.where(kernel==255)
    #print("a",x_un.shape,y_un.shape)
    x1=np.zeros([x_un.shape[0],x_un.shape[0]])
    y1=np.zeros([y_un.shape[0],y_un.shape[0]])
    x1[:,0]=x_un
    y1[:,0]=y_un
    x1[0,1:]=x_un[1:]
    y1[0,1:]=y_un[1:]
    branched=False
    #print(x1.shape,y1.shape)
    for j in range(1,x_un.shape[0]):
        res=np.zeros(x_un.shape[0]-1,dtype=int)
        res[:j]=x_un[:j]
        if j+1<x_un.shape[0]:
            #print("z")
            res[j:]=x_un[j+1:]
       # print("b",res.shape,x1.shape,x1[j,1:].shape)
        x1[j,1:]=np.copy(res)
    
        res[:j]=y_un[:j]
        if j+1<y_un.shape[0]:
            res[j:]=y_un[j+1:]
        #print("c",res.shape)
        y1[j,1:]=res
    x1=abs(x1-x1[:,0])
    y1=abs(y1-y1[:,0])
    x1=x1[:,1:]
    y1=y1[:,1:]
    #print("d",x_un.shape,y_un.shape)
    for j in range(x1.shape[0]):
        for k in range(x1.shape[1]):
            if x1[j,k]==2 or y1[j,k]==2 or (x1[j,k]==1 and y1[j,k]==1):
                branched=True
                break
                      
    if branched:
        branches_x=[]
        branches_y=[]
        for j in range(x_un.shape[0]):
            if abs(x_un[j]-y_un[j])==1:
                branches_x.append(x_un[j]+x-1)
                branches_y.append(y_un[j]+y-1)
            elif abs(x_un[j]-y_un[j])==0 and x_un[j]==0 and kernel[x_un[j]+1,y_un[j]]!=255 and kernel[x_un[j],y_un[j]+1]!=255:
                branches_x.append(x_un[j]+x-1)
                branches_y.append(y_un[j]+y-1)
            elif abs(x_un[j]-y_un[j])==0 and x_un[j]==2 and kernel[x_un[j]-1,y_un[j]]!=255 and kernel[x_un[j],y_un[j]-1]!=255:
                branches_x.append(x_un[j]+x-1)
                branches_y.append(y_un[j]+y-1)
            elif abs(x_un[j]-y_un[j])==2 and x_un[j]==2 and kernel[x_un[j]-1,y_un[j]]!=255 and kernel[x_un[j],y_un[j]+1]!=255:
                branches_x.append(x_un[j]+x-1)
                branches_y.append(y_un[j]+y-1)
            elif abs(x_un[j]-y_un[j])==2 and x_un[j]==0 and kernel[x_un[j]+1,y_un[j]]!=255 and kernel[x_un[j],y_un[j]-1]!=255:
                branches_x.append(x_un[j]+x-1)
                branches_y.append(y_un[j]+y-1)
                
                
        return True,branches_x,branches_y
    else:
        diff=abs(x_un-y_un)
        t=np.where(diff==1)
        x_next=x_un[t]
        y_next=y_un[t]

        return False,x_next,y_next   
def which_branch(x_br,y_br,x,y,new_img,marked_img,mark1,length2,last_branch):                    
    m_img=marked_img
    n_img=new_img
    this_branch_length=length2
    
    #print(mark1)
    temp_mark=mark1
    length=[]
    branches=[]
    branches.append(last_branch)
    length.append(length2)
    for i in range(len(x)):
        x_s=x[i]
        y_s=y[i]
        m_img[x_s,y_s]=temp_mark
        m_img,l,branch_points=branch_mark(x_s,y_s,n_img,m_img,temp_mark)
        branches.append(branch_points)        
        length.append(l)
        
    ir=len(length)
    s_r=length.index(max(length))
    for i in range(2,ir):
        s_d=length.index(max(length))
        s_b=length.index(min(length))

        point=branches[s_b]
        x_s=point[0]
        y_s=point[1]
        for j in range(len(x_s)):
            if x_s[j]!=x_br or y_s[j]!=y_br:
                n_img[x_s[j],y_s[j]]=0
                m_img[x_s[j],y_s[j]]=0
        branches.remove(branches[s_b])
        #print(length)
        length.remove(length[s_b])
    s_g=length.index(max(length))   
    last_x=0
    last_y=0
    if s_r==0:           
        the_branch=branches[1]
        last_x=the_branch[0][-1]
        last_y=the_branch[1][-1]
    else:
        the_branch=branches[s_g]
        last_x=the_branch[0][-1]
        last_y=the_branch[1][-1]

    return n_img,m_img,last_x,last_y
def branch_mark(x_1,y_1,new_img,marked_img,mark1):
    x=x_1
    y=y_1
    length=0
    n_img=new_img
    m_img=marked_img
    m_img[x,y]=mark1
    end_point=0
    x_branches=[]
    y_branches=[]
    x_branches.append(x)
    y_branches.append(y)
    while end_point!=1:
        
        kern=(n_img[x-1:x+2,y-1:y+2])
        marked_kernel=(m_img[x-1:x+2,y-1:y+2])//mark1
        
       # print(kern)
        #print(marked_kernel)
    
        kern=kern-(marked_kernel*255)
        #print(kern)
        s=np.sum(kern)//255
        if s==1:
            
            x_next,y_next=np.where(kern==255)
            x_m=x+x_next[0]-1
            y_m=y+y_next[0]-1
            m_img[x_m,y_m]=mark1
            length+=1
            x=x_m
            y=y_m
            x_branches.append(x)
            y_branches.append(y)
            
        elif s>=2:
            b,x_b,y_b=branches(kern,x,y)
            
            if b==False:
                x_next=x_b[0]
                y_next=y_b[0]
                m_img[x_next+x-1,y+y_next-1]=mark1
                length+=1
                x=x_next+x-1
                y=y_next+y-1
                x_branches.append(x)
                y_branches.append(y)
            else:
                n_img,m_img,x_next,y_next=which_branch(x,y,x_b,y_b,n_img,m_img,mark1,length,[x_branches,y_branches])
                x=x_next
                y=y_next
                x_branches.append(x)
                y_branches.append(y)
        if s==0:
            
            end_point=1
            
            
    return m_img,length,[x_branches,y_branches]            

def mark(x_1,y_1,new_img,marked_img,mark1):
    x=x_1
    y=y_1
    #print(mark1)
    length=0
    n_img=new_img
    m_img=marked_img
    m_img[x,y]=mark1
    end_point=0
    x_branch=[]
    y_branch=[]
    x_branch.append(x)
    y_branch.append(y)
    while end_point!=1:
        
        kern=(n_img[x-1:x+2,y-1:y+2])
       # print(m_img[x-1:x+2,y-1:y+2])
        marked_kernel=(m_img[x-1:x+2,y-1:y+2])//mark1
        
        #print(kern)
        #print(marked_kernel)
        kern=kern-(marked_kernel*255)
        #print(kern)
        s=np.sum(kern)//255
        if s==1:
            
            x_next,y_next=np.where(kern==255)
            x_m=x+x_next[0]-1
            y_m=y+y_next[0]-1
            m_img[x_m,y_m]=mark1
            length+=1
            x=x_m
            y=y_m
            x_branch.append(x)
            y_branch.append(y)
        elif s>=2:
            b,x_b,y_b=branches(kern,x,y)
            
            if b==False:
                x_next=x_b[0]
                y_next=y_b[0]
                m_img[x_next+x-1,y+y_next-1]=mark1
                length+=1
                x=x_next+x-1
                y=y_next+y-1
                x_branch.append(x)
                y_branch.append(y)
            else:
                #print("branch")
                n_img,m_img,x_next,y_next=which_branch(x,y,x_b,y_b,n_img,m_img,mark1,length,[x_branch,y_branch])
                x=x_next
                y=y_next
        if s==0:
            end_point=1
            
            
    return n_img,m_img,length,x,y            


# In[16]:


def count_conq(x_end,y_end,new_img,count):
    mark1=[0]*len(x_end)
    marked_img=np.zeros(new_img.shape,dtype=np.uint8)
    mark2=1

    iterq=0
    for i in range(len(x_end)):
        if mark1[i]==0 and mark2<=255:
            x=x_end[i]
            y=y_end[i]
            marked_img[x,y]=mark2
            mark1[i]=mark2
            new_img,marked_img,length,x_l,y_l=mark(int(x),int(y),new_img,marked_img,mark2)
            for j in range(len(x_end)):
                p=x_end[j]
                if p==x_l and y_end[j]==y_l:
                    break
            mark1[j]=mark2
            mark2+=1
            iterq+=1 
    #if count%1==0:
        #cv2.imwrite( "C:/Users/Pranesh/Desktop/steps/removed"+str(count)+".png", new_img)
    return new_img,marked_img,mark1,mark2
    


# In[ ]:





# In[18]:


def last_calc(new_img,marked_img,mark1,mark2):
    x_marked=[]
    y_marked=[]
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            if marked_img[i,j]!=0:
                x_marked.append(i)
                y_marked.append(j)
    all_length=np.zeros((mark2,1),dtype=int)

    for i in range(1,mark2+1):
        length=0
        x=[]
        y=[]
        for j in range(len(x_marked)):
            if marked_img[x_marked[j],y_marked[j]]==i:
                length+=1
                x.append(x_marked[j])
                y.append(y_marked[j])
        all_length[i-1]=length
    #print(length)
        if length<=5:
            for j in range(len(x)):
                new_img[x[j],y[j]]=0
                marked_img[x[j],y[j]]=0
            all_length[i-1]=0
    points=np.zeros((mark2,4))
   # print(len(mark1),len(x_end),len(y_end))
    for j in range(1,mark2+1):
        c=0
        for i in range(min(len(x_end),len(mark1),len(y_end))):
            if mark1[i]==j and c==0:
                points[j-1,0]=x_end[i]
                points[j-1,1]=y_end[i]
                c=1
            elif mark1[i]==j and c==1:
                points[j-1,2]=x_end[i]
                points[j-1,3]=y_end[i]               
    difference=points[:,0:2]-points[:,2:]
    squared=difference*difference
    sq_sum=np.sum(squared,1)
    point_dist=np.sqrt(sq_sum)


    closed=np.zeros((mark2,1))
    for i in range(point_dist.shape[0]):
        if all_length[i]!=0 and point_dist[i]!=0:
            r=all_length[i]/point_dist[i]
            if r>=2.:
                closed[i]=5
            else:
                closed[i]=1
            
    all_marked_cat=[]
    for j in range(1,mark2+1):
        x=[]
        y=[]
        for i in range(len(x_marked)):
            if marked_img[x_marked[i],y_marked[i]]==j:
                x.append(x_marked[i])
                y.append(y_marked[i])
        all_marked_cat.append([x,y])     
    
    minimum=10000
    for i in range(all_length.shape[0]):
        if all_length[i]<minimum and all_length[i]!=0:
            minimum=all_length[i]

    compressed_img=np.zeros(new_img.shape,dtype=np.uint8)
    marked_comprsd=np.zeros(new_img.shape,dtype=np.uint8)
    for i in range(1,mark2+1):
        if all_length[i-1]!=0:
            p1=points[i-1,0:2]
            p2=points[i-1,2:]
            compressed_img[int(p1[0]),int(p1[1])]=255
            compressed_img[int(p2[0]),int(p2[1])]=255
            marked_img[int(p1[0]),int(p1[1])]=i
            marked_img[int(p2[0]),int(p2[1])]=1          
        
    for i in range(1,mark2+1):
        l=len(all_marked_cat[i-1][0])
        x=all_marked_cat[i-1][0]
        y=all_marked_cat[i-1][1]
        if len(x)>0 and all_length[i-1]!=0:
            skip=1
            count=0
            for j in range(0,len(x),int(skip)):
                if count<minimum:
                    count+=1
                    compressed_img[x[j],y[j]]=255
                    marked_comprsd[x[j],y[j]]=i


    marked_x=[]
    marked_y=[]
    for i in range(compressed_img.shape[0]):
        for j in range(compressed_img.shape[1]):
            if marked_comprsd[i,j]!=0:
                marked_x.append(i)
                marked_y.append(j)
            
    all_marked_com=[]
    for j in range(1,mark2+1):
        x=[]
        y=[]
        for i in range(len(marked_x)):
            if marked_comprsd[marked_x[i],marked_y[i]]==j:
                x.append(marked_x[i])
                y.append(marked_y[i])
        all_marked_com.append([x,y])
    
    
    
    theta_all=np.zeros([len(all_marked_com),1])
    theta=0.0
    the_min=np.zeros([len(all_marked_com),2])
    for i in range(points.shape[0]):
        if all_length[i]!=0:
            x1=points[i,0]
            y1=points[i,1]
            x2=points[i,2]
            y2=points[i,3]
        
            if y2!=y1:
                m=(x2-x1)/(y2-y1)
                theta=math.atan(m)
            else:
                theta=math.radians(np.sign(x2-x1)*90)
            theta_all[i]=theta   
    
    
    blank=np.zeros(new_img.shape,dtype=np.uint8)

    x=all_marked_com[0][0]
    y=all_marked_com[0][1]
    if len(x)!=0:
        for j in range(len(x)):
            blank[x[j],y[j]]=255

        
    all_marked_com_local=[]
    for i in range(len(all_marked_com)):
        x=all_marked_com[i][0]
        y=all_marked_com[i][1]
        x_l=[]
        y_l=[]
    
        for j in range(len(x)):
            xl=(x[j]-points[i,0])*math.cos(theta_all[i])-(y[j]-points[i,1])*math.sin(theta_all[i])
            yl=(x[j]-points[i,0])*math.sin(theta_all[i])+(y[j]-points[i,1])*math.cos(theta_all[i])
            x_l.append(int(xl))
            y_l.append(int(yl))
        all_marked_com_local.append([x_l,y_l])
    

    all_sum=np.zeros((len(all_marked_com_local),1))
    for i in range(len(all_marked_com_local)):
        s=sum(all_marked_com_local[i][0])
        all_sum[i]=s
    
    all_sign=np.sign(all_sum)
    all_degrees=np.abs(np.degrees(theta_all))

    for i in range(1,mark2+1):
        if closed[i-1]==1:
            x_ref=all_marked_com[i-1][0]
            y_ref=all_marked_com[i-1][1]
            if len(x_ref)>0 and all_length[i-1]!=0:
                x1=points[i-1,0]
                y1=points[i-1,1]
                x2=points[i-1,2]
                y2=points[i-1,3]
                sign1=all_sign[i-1]
                for j in range(1,mark2+1):
                    if i!=j and closed[j-1]==1 and all_length[j-1]!=0:
                        a1=points[j-1,0]
                        b1=points[j-1,1]
                        a2=points[j-1,2]
                        b2=points[j-1,3]
                        sign2=all_sign[j-1]
                        d1=math.sqrt((a1-x1)**2+(b1-y1)**2)
                        d2=math.sqrt((a2-x2)**2+(b2-y2)**2)
                        k=-1
                        if d1<all_length[i-1] and d2<all_length[i-1]:
                            if np.sign(theta_all[i-1])==np.sign(theta_all[j-1]) and sign1==(-sign2):
                                k=1
                            elif np.sign(theta_all[i-1])!=np.sign(theta_all[j-1]) and sign1==sign2:
                                if all_degrees[i-1]<=45:
                                    if min(b1,b2)<min(y1,y2) and sign1>0:
                                        k=1
                                    elif max(b1,b2)>max(y1,y2) and sign1<0:
                                        k=1
                                elif all_degrees[i-1]>45:
                                    if max(a1,a2)>max(x1,x2) and sign1>0:
                                        k=1
                                    elif min(a1,a2)<min(x1,x2) and sign1<0:
                                        k=1
                            elif np.sign(theta_all[i-1])!=np.sign(theta_all[j-1]) and sign1==(-sign2):
                                if all_degrees[i-1]<=45:
                                    if max(b1,b2)>max(y1,y2) and sign1>0:
                                        k=1
                                    elif min(b1,b2)<min(y1,y2) and sign1<0:
                                        k=1
                                elif all_degrees[i-1]>45:
                                    if min(a1,a2)<min(x1,x2) and sign1>0:
                                        k=1
                                    elif max(a1,a2)>max(x1,x2) and sign1<0:
                                        k=1
                        if k==1:
                            x=all_marked_com[j-1][0]
                            y=all_marked_com[j-1][1]
                            for z in range(len(x)):
                                marked_comprsd[x[z],y[z]]=i
                                all_marked_com[j-1]=[[],[]]
                        
    x_marked=[]
    y_marked=[]
    for i in range(marked_comprsd.shape[0]):
        for j in range(marked_comprsd.shape[1]):
            if marked_comprsd[i,j]!=0:
                x_marked.append(i)
                y_marked.append(j)

    all_marked_reg=[]
    for i in range(1,mark2+1):
        x=[]
        y=[]
        for j in range(len(x_marked)):
            if marked_comprsd[x_marked[j],y_marked[j]]==i:
                x.append(x_marked[j])
                y.append(y_marked[j])
        all_marked_reg.append([x,y])                  
          
    d=[]
    blank=np.ones(new_img.shape,dtype=np.uint8)*255
    for i in range(len(all_marked_reg)):
        x=all_marked_reg[i][0]
        y=all_marked_reg[i][1]
        if len(x)>=5:
            coord=np.zeros((len(x),2),dtype='float32')
            coord[:,0]=y
            coord[:,1]=x
            ellipse=cv2.fitEllipse(coord)
            d.append(ellipse)
        
    sphericity=[]
    area=[]
    perimeter=[]
    r=[]
    for i in d:
        a=(3.14*i[1][0]*i[1][1]/4.)
        A=i[1][1]/2
        B=i[1][0]/2
        m1=math.sqrt((-3*((A-B)**2)/(A+B)**2)+4)
        p=3.14*(A+B)*(((3*(A-B)**2)/((A+B)**2*(m1+10)))+1)
        s=0
        if i[1][0]!=0:
            s=i[1][0]/i[1][1]
        else:
            s=0
        area.append(a)
        perimeter.append(p)
        sphericity.append(s)
    
    
    
        r1=2*a/(p*15)
        
        r.append(r1)
    
    volume=[]
    for i in range(len(r)):
        volume.append(4*3.14*(r[i])*(r[i])*r[i]/3)
    
    return r,sphericity,volume 


# In[56]:


count=200
number=[]
sph_big=[]
max_r=[]
mean_r=[]
void_all=[]
r1=0
r2=0
r3=0
r4=0
r5=0
r6=0
r7=0
r8=0
r9=0
r10=0
r11=0
r12=0
for i in range(0,count,20):
    sum_r=0
    sum_sph=0
    sum_volume=0
    l=0
    img0=cv2.imread( "C:/Users/Pranesh/Desktop/vid8/bubbles"+str(i)+".png",0)
    all_frames=np.ones((896,896,50))
    for j in range(1,50):
        all_frames[:,:,j]=cv2.imread( "C:/Users/Pranesh/Desktop/vid8/bubbles"+str(j+i+np.random.randint(1,20)*j)+".png",0)
        t=all_frames[:,:,j].astype('uint8')
        all_frames[:,:,j]=cv2.bilateralFilter(t,9,75,75)
    img0=img0.astype('uint8')
    all_frames[:,:,0]=cv2.bilateralFilter(img0,9,75,75)
    #imshow(img0)
    
    x_b,y_b,v=segment(all_frames,i)
    #imshow(v)
    
    new_img=Thinned(v,x_b,y_b,i)
    x_end,y_end,new_imga=ending(new_img)
    new_img2,marked_img,mark1,mark2=count_conq(x_end,y_end,new_imga,i)
    #print(len(r),len(sphericity),len(volume))
    r,sphericity,volume=last_calc(new_img2,marked_img,mark1,mark2)
    rd=32*16/15
    sv=sum(volume)
    vd=sv/(3.14*rd*rd*new_img.shape[0]//15)
    if vd<1:
        for j in range(min(len(r),len(volume),len(sphericity))):
            if r[j]>0.5:
                sum_r=sum_r+r[j]
                sum_sph=sum_sph+sphericity[j]
                sum_volume=sum_volume+volume[j]
                l=l+1
                if j<=3 and j>0.5:
                    r1+=1
                elif j<=6 and j>3:
                    r2+=1
                elif j<=9 and j>6:
                    r3+=1
                elif j<=12 and j>9:
                    r4+=1
                elif j<=15 and j>12:
                    r5+=1
                elif j<=18 and j>15:
                    r6+=1
                elif j>18 and j<=21:
                    r7+=1
                elif j>21 and j<=24:
                    r8+=1
                elif j>24 and j<=30:
                    r9+=1
                elif j>30 and j<=40:
                    r10+=1
                elif j>40 and j<=55:
                    r11+=1
                elif j>55:
                    r12+=1
        if l==0:
            l=1
        mn_r=sum_r/l
        mn_sph=sum_sph/l
        void=sum_volume/(3.14*rd*rd*new_img.shape[0]//11)
        for m in range(len(r)):
            if r[m]<=0.01:
                r[m]=0
        if len(r)>=1:
            number.append(l)
            sph_big.append(mn_sph)
        
            max_r.append(max(r))
            mean_r.append(mn_r)
            void_all.append(void)
        
        print(i)


# In[29]:


import xlwt 
from xlwt import Workbook 
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')


# In[59]:


#sheet1.write( 0,42,"8")
for i in range(1,len(mean_r)+1):
    #sheet1.write( i,42, number[i-1])
    sheet1.write( i,43 ,void_all[i-1])
    sheet1.write( i,44, mean_r[i-1])
    sheet1.write( i,45, max_r[i-1])
    sheet1.write( i,46, sph_big[i-1])


# In[54]:


sheet1.write( 8 ,40, r1)
sheet1.write( 9 ,40, r2)
sheet1.write( 10,40, r3)
sheet1.write( 11,40, r4)
sheet1.write( 12,40, r5)
sheet1.write( 13,40, r6)
sheet1.write( 14,40, r7)
sheet1.write( 15,40, r8)
sheet1.write( 16,40, r9)
sheet1.write( 17,40, r10)
sheet1.write( 18,40, r11)
sheet1.write( 19,40, r12)




# In[60]:


wb.save('C:/Users/Pranesh/Desktop/my_data.xls') 


