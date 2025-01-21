#install.packages("ggplot2")
library(ggplot2)

#load scv filasdasd asd e
df = read.csv("Mosquito_Supermodel/src/tests/csvs/visits_vs_frames.csv")
df = df[df$time_interval<=20,]
#plot it
plot = ggplot(df, aes(x = time_interval, y = mean_trajectory_count, color = teratment_or_rep, group = teratment_or_rep)) +
  geom_point() +  # Add points
  geom_line() +  # Add lines
  geom_errorbar(aes(ymin = mean_trajectory_count - se_trajectory_count,
                    ymax = mean_trajectory_count + se_trajectory_count), width = 0.2) +  # Add error bars
  labs(title = " ",
       x = "Time (minutes)", y = "Mean Trajectory Count (±SE)", color = "Treatment") 
       #save it
ggsave("Mosquito_Supermodel/src/r/sdfsd.jpeg", plot = plot, width = 8, height = 6)
