#install.packages("ggplot2")
library(ggplot2)

#load scv filasdasd asd e
df = "src\\tests\\plots\\average_visits.csv"
#plot it
 p =ggboxplot(df, "image_name", "trajectory_count",
    add = "jitter")

p
#save it
ggsave("src/playground/sdfsd.png")