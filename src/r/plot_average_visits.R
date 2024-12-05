library("ggpubr")
library("ggplot2")

#load scv filasdasd asd e

#plot it
p = ggplot(df[df$treatment==i &df$sex==sex,], aes(x = rep, y = NC,fill = as.factor(rep))) +
    geom_boxplot() + geom_point() + 
    scale_y_continuous(breaks = seq(0,1600,by = 400),limits =c(0,1600),expand = c(0,0))+
    labs(title = paste("tretment is",i,"\n","sex is",sex), x = "rep", y = "Normelized trajectory counts") +
    theme_classic() +theme(legend.position = "none") 


#save it
ggsave("src/playground/sdfsd.png")