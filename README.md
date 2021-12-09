# Pneumonia Classification

This is a simple `REST` api that is served to classify `pneumonia` given an X-ray image of a chest of a human being. The following are expected results when the model does it's classification.

1. pneumonia bacteria
2. pneumonia virus
3. normal

<p align="center" with="100%"><img src="/images/pneumonia.jpg" width="100%" alt=""/>
</p>

### Starting the server

To run this server and make prediction on your own images follow the following steps

1. create a virtual environment and activate it
2. run the following command to install packages

```shell
pip install -r requirements.txt
```

3. navigate to the `app.py` file and run

```shell
python app.py
```

### Model

We are using a simple **M**ulti **L**ayer **P**erceptron **(MLP)** achitecture to do the categorical image classification on chest-x-ray images which looks simply as follows:

```py
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=.5):
        super(MLP, self).__init__()
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.input_fc(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_fc(x))
        x = self.dropout(x)
        outputs = self.output_fc(x)
        return outputs, x
```

> All images are transformed to `grayscale`.

### Model Metrics

The following table shows all the metrics summary we get after training the model for few `10` epochs.

<table border="1">
    <thead>
      <tr>
        <th>model name</th>
        <th>model description</th>
        <th>test accuracy</th>
        <th>validation accuracy</th>
        <th>train accuracy</th>
         <th>test loss</th>
        <th>validation loss</th>
        <th>train loss</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>chest-x-ray.pt</td>
        <td>pneumonia classification using Multi Layer Perceprton (MLP)</td>
        <td>73.73%</td>
        <td>73.73%</td>
        <td>72.47%</td>
        <td>0.621</td>
        <td>0.621</td>
        <td>0.639</td>
      </tr>
       </tbody>
  </table>

### Classification report

This classification report is based on the first batch of the test dataset i used which consist of `64` images in a batch.

<table border="1">
    <thead>
      <tr>
        <th>#</th>
        <th>precision</th>
        <th>recall</th>
        <th>f1-score</th>
        <th>support</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>micro avg</td>
        <td>100%</td>
        <td>81%</td>
        <td>90%</td>
        <td>4096</td>
      </tr>
      <tr>
        <td>macro avg</td>
        <td>100%</td>
        <td>81%</td>
        <td>90%</td>
        <td>4096</td>
      </tr>
      <tr>
        <td>weighted avg</td>
        <td>100%</td>
        <td>81%</td>
        <td>90%</td>
        <td>4096</td>
      </tr>
    </tbody>
  </table>

### Confusion matrix

The following image represents a confusion matrix for the first batch in the validation set which contains `64` images in a batch:

<p align="center" with="100%"><img src="images/cm.png" width="100%" alt=""/>
</p>

### Pneumonia classification

If you hit the server at `http://localhost:3001/api/pneumonia` you will be able to get the following expected response that is if the request method is `POST` and you provide the file expected by the server.

### Expected Response

The expected response at `http://localhost:3001/api/pneumonia` with a file `image` of the right format will yield the following `json` response to the client.

```json
{
  "predictions": {
    "class_label": "PNEUMONIA VIRAL",
    "label": 2,
    "meta": {
      "description": "given a medical chest-x-ray image of a human being we are going to classify weather a person have pneumonia virus, pneumonia bacteria or none of those(normal).",
      "language": "python",
      "library": "pytorch",
      "main": "computer vision (cv)",
      "programmer": "@crispengari"
    },
    "predictions": [
      {
        "class_label": "NORMAL",
        "label": 0,
        "probability": 0.15000000596046448
      },
      {
        "class_label": "PNEUMONIA BACTERIA",
        "label": 1,
        "probability": 0.10000000149011612
      },
      { "class_label": "PNEUMONIA VIRAL", "label": 2, "probability": 0.75 }
    ],
    "probability": 0.75
  },
  "success": true
}
```

### Using `curl`

Make sure that you have the image named `normal.jpeg` in the current folder that you are running your `cmd` otherwise you have to provide an absolute or relative path to the image.

> To make a `curl` `POST` request at `http://localhost:3001/api/pneumonia` with the file `normal.jpeg` we run the following command.

```shell
curl -X POST -F image=@normal.jpeg http://127.0.0.1:3001/api/pneumonia
```

### Using Postman client

To make this request with postman we do it as follows:

1. Change the request method to `POST`
2. Click on `form-data`
3. Select type to be `file` on the `KEY` attribute
4. For the `KEY` type `image` and select the image you want to predict under `value`
5. Click send

If everything went well you will get the following response depending on the face you have selected:

```json
{
  "predictions": {
    "class_label": "NORMAL",
    "label": 0,
    "meta": {
      "description": "given a medical chest-x-ray image of a human being we are going to classify weather a person have pneumonia virus, pneumonia bacteria or none of those(normal).",
      "language": "python",
      "library": "pytorch",
      "main": "computer vision (cv)",
      "programmer": "@crispengari"
    },
    "predictions": [
      {
        "class_label": "NORMAL",
        "label": 0,
        "probability": 0.8500000238418579
      },
      {
        "class_label": "PNEUMONIA BACTERIA",
        "label": 1,
        "probability": 0.07000000029802322
      },
      {
        "class_label": "PNEUMONIA VIRAL",
        "label": 2,
        "probability": 0.07999999821186066
      }
    ],
    "probability": 0.8500000238418579
  },
  "success": true
}
```

### Using JavaScript `fetch` api.

1. First you need to get the input from `html`
2. Create a `formData` object
3. make a POST requests

```js
const input = document.getElementById("input").files[0];
let formData = new FormData();
formData.append("image", input);
fetch("http://127.0.0.1:3001/api/pneumonia", {
  method: "POST",
  body: formData,
})
  .then((res) => res.json())
  .then((data) => console.log(data));
```

If everything went well you will be able to get expected response.

```json
{
  "predictions": {
    "class_label": "PNEUMONIA VIRAL",
    "label": 2,
    "meta": {
      "description": "given a medical chest-x-ray image of a human being we are going to classify weather a person have pneumonia virus, pneumonia bacteria or none of those(normal).",
      "language": "python",
      "library": "pytorch",
      "main": "computer vision (cv)",
      "programmer": "@crispengari"
    },
    "predictions": [
      {
        "class_label": "NORMAL",
        "label": 0,
        "probability": 0.15000000596046448
      },
      {
        "class_label": "PNEUMONIA BACTERIA",
        "label": 1,
        "probability": 0.10000000149011612
      },
      { "class_label": "PNEUMONIA VIRAL", "label": 2, "probability": 0.75 }
    ],
    "probability": 0.75
  },
  "success": true
}
```

### Notebooks

The `ipynb` notebook that i used for training the model and saving an `.pt` file was can be found:

1. [Model Training And Saving](https://github.com/CrispenGari/cv-torch/blob/main/01-PNEUMONIA-CLASSIFICATION/01_MLP_Pneumonia.ipynb)
