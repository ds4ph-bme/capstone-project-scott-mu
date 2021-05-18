library(shiny)
library(reticulate)
library(tidyverse)
library(keras)
library(nhanesA)
library(tensorflow)
library(tfdatasets)
library(rsample)
library(lime)
library(recipes)
library(pROC)
library(corrr)



#Adding prediction functions to enable LIME predictions
model_type.keras.engine.sequential.Sequential  <<- function(x, ...) {
  return("classification")
}
predict_model.keras.engine.sequential.Sequential <<- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  return(data.frame(Yes = pred, No = 1 - pred))
}


ui <- fluidPage(
    titlePanel("Deep Learning Predictions of Self-Reported Diabetes"),
    h4("A Keras/TensorFlow Multilayer Perceptron Model with NHANES Anthropometric and Demographic Data"),
    sidebarLayout(
        sidebarPanel(
            p("This is an artificial neural network machine learning model built using information from 4,897 non-pregant adults in the 2017-2018 National Health and Nutrition Examination Survey with complete demographic, body measures, and self-reported diabetes data. Displayed to the right are several plots: training metrics, feature importance correlation in the training set, receiver operating characteristics (ROC) of testing set, and relative contribution each user-input variable to the final prediction. 80% of the data is used to train/validate the model and the remainder is used to test the model."),
            
            br(),
            
            selectInput("gender", "Gender:", choices = list("Male"=1, "Female" =2)),

            selectInput("race", "Race/Ethnicity:", choices = list(
                "Mexican American"=1, 
                "Other Hispanic" =2,
                "Non-Hispanic White" =3,
                "Non-Hispanic Black" =4,
                "Non-Hispanic Asian"=6,
                "Other Race - Including Multi-Racial" = 7)),
            sliderInput("age",
                        "Age (year):",
                        min = 18,
                        max = 80,
                        value = 50),
            sliderInput("weight",
                        "Weight (kg):",
                        min = 30,
                        max = 200,
                        value = 81),
            sliderInput("height",
                        "Height (cm):",
                        min = 120,
                        max = 220,
                        value = 166),
            sliderInput("leg",
                        "Upper Leg Length (cm):",
                        min = 25,
                        max = 60,
                        value = 40),
            sliderInput("armlen",
                        "Upper Arm Length (cm):",
                        min = 25,
                        max = 50,
                        value = 37),
            sliderInput("armcirc",
                        "Arm Circumference (cm):",
                        min = 15,
                        max = 60,
                        value = 32),
            sliderInput("waist",
                        "Waist Circumference (cm):",
                        min = 50,
                        max = 170,
                        value = 100),
            sliderInput("hip",
                        "Hip Circumference (cm):",
                        min = 70,
                        max = 180,
                        value = 105)

        )
        
        ,
        mainPanel(

          wellPanel(h3(textOutput("predictedprobability"))),
          
          #Testing
     #     tableOutput("baztable"),
          
          
          splitLayout(
          wellPanel(plotOutput("limeplot")),
          wellPanel(plotOutput("metricplot"))
          )
          ,
          
          splitLayout(
          wellPanel(plotOutput("corrrplot")),
          wellPanel(plotOutput("rocplot"))
           
          )
           
           
        )
    )
)

server <- function(input, output) {
    
    #Downloading NHANES data
    demo_j = nhanes("DEMO_J")
    bmx_j = nhanes("BMX_J")
    diq_j = nhanes("DIQ_J")
    
    #Merging 3 data sets
    dat = merge(merge(demo_j, bmx_j), diq_j)
    
    #Excluding pregnant, children, and missing/inconclusive data
    d = dat %>% 
        filter(RIDAGEYR >=18, is.na(RIDEXPRG)|RIDEXPRG==2|RIDEXPRG==3, DIQ010==1|DIQ010==2) %>% 
        select(DIQ010, RIAGENDR, RIDAGEYR, RIDRETH3, BMXWT, BMXHT, BMXLEG, BMXARML, BMXARMC, BMXWAIST, BMXHIP) %>%
        drop_na() %>% 
        mutate(across(c(RIAGENDR, RIDRETH3), factor))
    
    #80-20 Testing/Training split
    train_test_split <- initial_split(d, prop = 0.8)
    train_tbl <- training(train_test_split)
    test_tbl  <- testing(train_test_split) 
    
    #Creating a recipe (one-hot encoding and standardizing variables)
    rec_obj = recipe(DIQ010 ~.  ,data=train_tbl) %>% 
        step_dummy(RIAGENDR, RIDRETH3, -all_outcomes()) %>%
        step_center(all_numeric(), -all_outcomes()) %>%
        step_scale(all_numeric(), -all_outcomes()) %>%
        prep(data = train_tbl)
    
    #Applying recipe to finish processing dataset
    x_train_tbl <- bake(rec_obj, new_data = train_tbl) %>% select(-DIQ010)
    x_test_tbl  <- bake(rec_obj, new_data = test_tbl) %>% select(-DIQ010)
    
    #Storing 
    y_train_vec <- ifelse(pull(train_tbl, DIQ010) == "1", 1, 0)
    y_test_vec  <- ifelse(pull(test_tbl, DIQ010) == "1", 1, 0)
    
    #Building the Neural Network Model
    model_keras <- keras_model_sequential()
    model_keras %>% 
        # First hidden layer
        layer_dense(
            units              = 16, 
            kernel_initializer = "uniform", 
            activation         = "relu", 
            input_shape        = ncol(x_train_tbl)) %>% 
        # Dropout to prevent overfitting
        layer_dropout(rate = 0.1) %>%
        # Second hidden layer
        layer_dense(
            units              = 16, 
            kernel_initializer = "uniform", 
            activation         = "relu") %>% 

        # Dropout to prevent overfitting
        layer_dropout(rate = 0.1) %>%
        # Output layer
        layer_dense(
            units              = 1, 
            kernel_initializer = "uniform", 
            activation         = "sigmoid") %>% 
        # Compile ANN
        compile(
            optimizer = 'adam',
            loss      = 'binary_crossentropy',
            metrics   = c('accuracy')
        )
    #keras_model
    history <- fit(
        object           = model_keras, 
        x                = as.matrix(x_train_tbl), 
        y                = y_train_vec,
        batch_size       = 10, 
        class_weight = list("0"=1,"1"=6),
        epochs           = 20,
        validation_split = 0.20,
        view_metrics=F
    )
    
    #Model metric plot
    output$metricplot = renderPlot( 
      plot(history, smooth=T) + theme_minimal()+
      labs(title="Training Model Metrics",
      x="Epoch",
       y=NULL))
    
    #Adding prediction functions to enable LIME predictions
    model_type.keras.engine.sequential.Sequential  <<- function(x, ...) {
      return("classification")
    }
    predict_model.keras.engine.sequential.Sequential <<- function(x, newdata, type, ...) {
      pred <- predict_proba(object = x, x = as.matrix(newdata))
      return(data.frame(Yes = pred, No = 1 - pred))
    }

    #Using training data to set local linear explanations
    explainer <- lime(
      x              = x_train_tbl, 
      model          = as_classifier(model_keras), 
      bin_continuous = FALSE
    )
    

 
    #Correlation analysis plot
    corrr_analysis <- x_train_tbl  %>% 
        rename(Age = "RIDAGEYR"   ,
               Weight = "BMXWT"     ,
               Height = "BMXHT"    ,
               `Leg length`= "BMXLEG"    ,
               `Arm length`="BMXARML"  ,
               `Arm circumference` = "BMXARMC"   ,
               `Waist circumference`="BMXWAIST"   ,
               `Hip circumference`="BMXHIP"    ,
               `Female`="RIAGENDR_X2", 
               `Other Hispanic`="RIDRETH3_X2" ,
               `Non-Hispanic White`="RIDRETH3_X3",
               `Non-Hispanic Black`="RIDRETH3_X4",
               `Non-Hispanic Asian`="RIDRETH3_X6" ,
               `Other Race`="RIDRETH3_X7") %>% 
                mutate(predprobs = y_train_vec) %>%
        correlate() %>%
        focus(predprobs)%>%
        rename(feature = term) %>%
        arrange(abs(predprobs)) %>%
        mutate(feature = as_factor(feature)) 
    
    #Correlation analysis plot
        corrr_plot = corrr_analysis %>%
        ggplot(aes(x = predprobs, y = fct_reorder(feature, desc(predprobs)))) +
        geom_point() +
        geom_segment(aes(xend = 0, yend = feature), 
                     color = "red", 
                     data = corrr_analysis %>% filter(predprobs > 0)) +
        geom_point(color = "red", 
                   data = corrr_analysis %>% filter(predprobs > 0)) +
        geom_segment(aes(xend = 0, yend = feature), 
                     color = "black", 
                     data = corrr_analysis %>% filter(predprobs < 0)) +
        geom_point(color = "black", 
                   data = corrr_analysis %>% filter(predprobs < 0)) +
        geom_vline(xintercept = 0, color = "black", alpha=0.3, size = 1, linetype = 1) +
        theme_minimal() +
        labs(title = "Training Data Correlation Analysis",
             subtitle = "Positive correlations (red) are globally predictive of diabetes",
             y=NULL,
             x = "Feature Correlation")

        output$corrrplot = renderPlot(corrr_plot)   
        
        
      #Construction ROC
        roc = roc(response =y_test_vec ,
                predictor=predict_model(model_keras, x_test_tbl)[,1], 
                ci=T)
        roc_ci = data.frame(ci.se(roc, specificities = seq(0,1, l=50)))
        dat.ci = data.frame(x = as.numeric(rownames(roc_ci)),
                             lower = roc_ci[, 1],
                             upper = roc_ci[, 3])
      #ROC gg_plot  
      roc_plot = ggroc(roc)+
            theme_minimal()+
            coord_fixed()+
            labs(x="Specificity", y="Sensitivity", 
                 title="Testing Set ROC",
                 subtitle=paste0("ROC=",round(roc$ci[2],2),
                                 ", ",
                                 "95% CI: ",
                                 round(roc$ci[1],2), " to ",
                                 round(roc$ci[3],2)
                                 ))+
            geom_ribbon(
                data = dat.ci,
                aes(x = x, ymin = lower, ymax = upper),
                fill = "steelblue",
                alpha = 0.2
            )
            
    #Rendering ROC plot
   output$rocplot = renderPlot(roc_plot)
   
   explanation_baz = reactive(
     data.frame(
       "RIAGENDR"= factor(input$gender),
       "RIDAGEYR" = input$age,
       "RIDRETH3" = factor(input$race),
       "BMXWT"  =input$weight,
       "BMXHT" =input$height,
       "BMXLEG" =input$leg,
       "BMXARML" =input$armlen,
       "BMXARMC" =input$armcirc,
       "BMXWAIST"=input$waist,
       "BMXHIP"=input$hip)
     
     )
   
   
   
   
   output$baztable=renderTable(explanation_baz())
   
   
   
   
   #Overall predicted model probability
   output$predictedprobability = renderText({
     
     newindividual_data =   data.frame(
       "RIAGENDR"= factor(input$gender),
       "RIDAGEYR" = input$age,
       "RIDRETH3" = factor(input$race),
       "BMXWT"  =input$weight,
       "BMXHT" =input$height,
       "BMXLEG" =input$leg,
       "BMXARML" =input$armlen,
       "BMXARMC" =input$armcirc,
       "BMXWAIST"=input$waist,
       "BMXHIP"=input$hip)
     
     x_newindividual_tbl  <- bake(rec_obj, new_data = newindividual_data) 
     
     paste0("Predicted Probability of Diabetes: ",round(predict_model(model_keras,x_newindividual_tbl)$Yes,2)*100,"%")
   })
   
        
    #LIME plot  
  output$limeplot=renderPlot({
  
  #Adding prediction functions to enable LIME
  model_type.keras.engine.sequential.Sequential  <- function(x, ...) {
    return("classification")
  }
  predict_model.keras.engine.sequential.Sequential <- function(x, newdata, type, ...) {
    pred <- predict_proba(object = x, x = as.matrix(newdata))
    return(data.frame(Yes = pred, No = 1 - pred))
  }
  
  if(1){
    #Reactive input into data frame
    newindividual_data =   data.frame(
      #"DIQ010"= NULL,
        "RIAGENDR"= factor(input$gender),
        "RIDAGEYR" = input$age,
        "RIDRETH3" = factor(input$race),
        "BMXWT"  =input$weight,
        "BMXHT" =input$height,
        "BMXLEG" =input$leg,
        "BMXARML" =input$armlen,
        "BMXARMC" =input$armcirc,
        "BMXWAIST"=input$waist,
        "BMXHIP"=input$hip)
    
  }
  
  #For debugging
    if(0){
    newindividual_data =   data.frame(
      #"DIQ010"= NULL,
      "RIAGENDR"= factor(2),
      "RIDAGEYR" = 25,
      "RIDRETH3" = factor(2),
      "BMXWT"  =25,
      "BMXHT" =25,
      "BMXLEG" =25,
      "BMXARML" =25,
      "BMXARMC" =25,
      "BMXWAIST"=25,
      "BMXHIP"=25)
    
    }
    

    #Process using original recipe
    x_newindividual_tbl  <- bake(rec_obj, new_data = newindividual_data) 

    #Create explanation for user input values
    if(0){
   explanation <- {
     
     model_type.keras.engine.sequential.Sequential  <- function(x, ...) {
       return("classification")
     }
     predict_model.keras.engine.sequential.Sequential <- function(x, newdata, type, ...) {
       pred <- predict_proba(object = x, x = as.matrix(newdata))
       return(data.frame(Yes = pred, No = 1 - pred))
     }
     
     predict_model.keras.engine.training.Model=  function(x, newdata, type, ...) {
       pred <- predict_proba(object = x, x = as.matrix(newdata))
       
       return(data.frame(Yes = pred, No = 1 - pred))
     }
       
       
       
     lime::explain(
       x_newindividual_tbl, 
       explainer    = explainer, 
       n_labels     = 1, 
       n_features   = 10,
       kernel_width = 0.5)
   }
   
  }
   
   #explain()
   explanation = lime::explain(
     x_newindividual_tbl, 
     explainer    = explainer, 
     n_labels     = 1, 
     n_features   = 10,
     kernel_width = 0.5)
   
   
   #Plotting explanation
   plot_features(explanation) +
       labs(title = "LIME Feature Importance Visualization",
            subtitle = "(Blue features are locally supportive of the label, red features locally contradict the label)")+
       labs(x=NULL)+
       theme(legend.position="none")
    
    })
    




}

# Run the application 
shinyApp(ui = ui, server = server)
