library(data.table)
library(huxtable)
if (FALSE){
eps = seq(.01,.15,by=.02)
delta = seq(.002,.01,by=.002)
df <- expand.grid(eps, delta)
setnames(df, c("eps","delta"))
df$m =  log(1/df$delta)/log(1/(1-df$eps))
df$m2 = ceiling((1/(2*(df$eps^2)))*log(1/df$delta))
df$theor = sqrt((1/(2*(df$m-1)))*log(1/df$delta))-sqrt((1/(2*df$m))*log(1/df$delta))
df$thresh =  df$eps - sqrt((1/(2*df$m2))*log(1/df$delta))
df$inverse = 1/df$thresh
xyplot(m~eps,groups=delta,df,type="l",
       main = "m(\U03B5) for different confidence levels",
       auto.key=list(title="\U03B4", space = "right", cex=1.0, just = 0.95))
eps = seq(.01,.3,by=.02)
delta = seq(.0002,.001,by=.0002)
m = seq(1,101,by=5)
df2 = expand.grid(eps,delta,m)
setnames(df2, c("eps","delta","m"))

df2$diff = sqrt((1/(2*df2$m))*log(1/df2$delta))
df2$thresh = df2$eps - sqrt((1/(2*df2$m))*log(1/df2$delta))
df2$ratio = 1/df2$thresh
head(df2[df2$thresh>0,])
}

model_names = c("monotonic",
                "robust_monotonic",
                "robust_combine_two",
                "robust_combine_three",
                "train_adv_combine",
                "baseline")
verbose_names = c("Monotonic XgBoost Model",
                  "Verifiably Robust (D)",
                  "Verifiably Robust (A+B)",
                  "Verifiably Robust (A+B+E)",
                  "Adversarially Trained (A+B)",
                  "Baseline Neural Network")
trainOptions = c("_train","")
distOptions = c("_uniform_edge","_uniform","_centered_edge","_empirical_edge")
grid = expand.grid(distOptions,trainOptions,model_names)
grid$filename = paste0(grid$Var3,grid$Var1,grid$Var2)
grid$filepath =paste0("~/code/pdfclassifier/train/tests/",grid$filename,".csv")
grid$trainString = ifelse(grid$Var2=="_train"," (Train)"," (Test)")
grid[grid$Var1=="_uniform_edge" | grid$Var1=="_uniform",]$trainString = ""
gridTest = grid[grid$Var2 == "",]
gridTrain = grid[grid$Var2 == "_train",]
gridTrain = gridTrain[gridTrain$Var1!="_uniform_edge" & gridTrain$Var1!="_uniform",]
grid = rbind(gridTest,gridTrain)
grid$distString = ""
grid[grid$Var1=="_uniform_edge",]$distString = "Edge on Hypercube"
grid[grid$Var1=="_uniform",]$distString = "Path on Hypercube"

grid[grid$Var1=="_centered_edge",]$distString = "Mutation"
grid[grid$Var1=="_empirical_edge",]$distString = "Empirical"
vb = data.table(model_name=model_names,verbose_name=(verbose_names))
grid = merge(grid,vb,by.x ="Var3",by.y= "model_name")
grid$caption = paste0(grid$verbose_name,": ",grid$pathString,grid$distString,grid$trainString)
filenames = grid$filename
filepaths = grid$filepath
captions = grid$caption

for (i in 1:length(filenames)){
  print(captions[i])
}

for (i in 1:length(filenames)){
  file=filepaths[i]
  caption = captions[i]
  print(caption)
  dat = read.csv(file)
  eps = unique(dat$epsilon)
  delta = unique(dat$delta)
  n_e = length(eps)
  n_d = length(delta)
  #M = matrix(0,ncol = n_d,nrow=n_e)
  M = matrix(NA,ncol = n_d,nrow=n_e)
  
  row.names(M)=eps
  colnames(M)=delta
  n = 3514
  
  for (i in 1:nrow(dat)){
    e_ind = which(eps==dat[i,]$epsilon)
    d_ind = which(delta==dat[i,]$delta)
    e = dat[i,]$epsilon
    d = dat[i,]$delta
    m_local = log(1/d)/log(n/(n-e))
    m_local = formatC(m_local, format = "e", digits = 0)
    if (dat[i,]$success=="Reject"){
      #M[e_ind,d_ind]=-1
      #M[e_ind,d_ind]=paste0("Rej. (m~",m_local,")")
      M[e_ind,d_ind]=paste0("Reject")
      
    }
    if (dat[i,]$success=="Accept"){
      #M[e_ind,d_ind]=1
      #M[e_ind,d_ind]=paste0("Accept (m~",m_local,")")
      M[e_ind,d_ind]=paste0("Accept")
    }
    if (dat[i,]$success=="N/A"){
      #M[e_ind,d_ind]=0
      M[e_ind,d_ind]=paste0("N/A (m~", m_local,")")
      M[e_ind,d_ind]=paste0("N/A")
    }
  }
  # kable(M, "latex")
  ht = as_hux(M,
              add_colnames=TRUE,
              add_rownames="\\epsilon \\hspace{4pt} $\\backslash$ \\delta",
              autoformat = getOption("huxtable.autoformat", TRUE),
              )
  width(ht) <- .7
  ht = set_caption(ht, caption)
  ht = set_escape_contents(ht,FALSE)
  for (i in 1:nrow(M)+1){
    for (j in 1:ncol(M)+1){
      if (grepl("Acc",ht[i,j])){
        ht=set_background_color(ht, i,j, "green")
      }
      if (grepl("Rej",ht[i,j])){
        ht=set_background_color(ht, i,j, "red")
      }
      if (grepl("N/A",ht[i,j])){
        ht=set_background_color(ht, i,j, "gray")
      }
    }
  }
  capture.output(print_latex(ht,tabular_only = F), file = '~/code/pdfclassifier/train/table.tex',append=TRUE)
  #heatmap.2(M,Rowv=FALSE, Colv=FALSE,na.color = "gray",dendrogram='none', "Adversarially Retrained Model Monotonicity")
}

