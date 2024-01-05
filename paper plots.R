library(data.table)
library(lattice)
library(gplots)
library(kableExtra)
library(huxtable)
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

file = "~/code/pdfclassifier/train/monotonic.csv"
dat = read.csv(file)
eps = unique(dat$epsilon)
delta = unique(dat$delta)
n_e = length(eps)
n_d = length(delta)
#M = matrix(0,ncol = n_d,nrow=n_e)
M = matrix(NA,ncol = n_d,nrow=n_e)

row.names(M)=eps
colnames(M)=delta

for (i in 1:nrow(dat)){
  e_ind = which(eps==dat[i,]$epsilon)
  d_ind = which(delta==dat[i,]$delta)
  print("success")
  print(dat[i,]$success)
  if (dat[i,]$success=="Reject"){
    #M[e_ind,d_ind]=-1
    M[e_ind,d_ind]="Reject"
    
  }
  if (dat[i,]$success=="Accept"){
    #M[e_ind,d_ind]=1
    M[e_ind,d_ind]="Accept"
  }
  if (dat[i,]$success=="N/A"){
    #M[e_ind,d_ind]=0
    M[e_ind,d_ind]="N/A"
  }
}
kable(M, "latex")
ht = as_hux(M,
            add_colnames=TRUE,
            add_rownames="\\epsilon \\ \\delta",
            autoformat = getOption("huxtable.autoformat", TRUE),
            caption = "Hello"
            )
set_caption(ht, "Adversarially Retrained Model Monotonicity")

for (i in 1:nrow(M)+1){
  for (j in 1:ncol(M)+1){
    if (ht[i,j]=="Accept"){
      ht=set_background_color(ht, i,j, "green")
    }
    if (ht[i,j]=="Reject"){
      ht=set_background_color(ht, i,j, "red")
    }
    if (ht[i,j]=="N/A"){
      ht=set_background_color(ht, i,j, "gray")
    }
  }
}
print_latex(ht)
heatmap.2(M,Rowv=FALSE, Colv=FALSE,na.color = "gray",dendrogram='none', "Adversarially Retrained Model Monotonicity")