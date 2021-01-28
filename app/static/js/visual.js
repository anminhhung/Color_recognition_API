
function initAllCam(camname){
    $.ajax({
        // url: 'http://service.aiclub.cs.uit.edu.vn/vehicles_counting/track_video?camname='+camname,
        url: '/track_video?camname='+camname,
        type: 'get',
        dataType: 'json',
        contentType: 'application/json',  
        success: function (response) {
            if (response['code'] == 1001) {
                alert("[Lỗi] Không nhận được phản hồi từ server, vui lòng kiểm tra lại!");
            }
        }
    }).done(function() {
        
    }).fail(function() {
        alert('Fail!');
    });
}

function drawImageOCR(src) {
    var canvas = document.getElementById("preview_img");
    console.log("drawImageOCR");
    IMGSRC = src;
    var context = canvas.getContext('2d');
    var imageObj = new Image();
    imageObj.onload = function() {
        canvas.width = this.width;
        canvas.height = this.height;
        context.drawImage(imageObj, 0, 0, this.width,this.height);
    };
    imageObj.src = src;
}

$('#btn-visual').click(function () {
    var camname = document.getElementById('list_cams').value;
    if (camname == 'cam1') {
        // document.getElementById("streamcam").src = "http://192.168.28.75:7778/stream1"
        document.getElementById("streamcam").src = "http://0.0.0.0:7778/stream1"
    } 
    else {
        // document.getElementById("streamcam").src = "http://192.168.28.75:7778/stream2"
        document.getElementById("streamcam").src = "http://0.0.0.0:7778/stream2"
    }
    
    console.log("camid: " + camname);
    initAllCam(camname);
    setInterval(getVehicleClass, 1000);
}
)

function getVehicleClass(){
    $.ajax({
        // url: 'http://service.aiclub.cs.uit.edu.vn/vehicles_counting/track_video?camname='+camname,
        url: '/vehicle_class',
        type: 'get',
        dataType: 'json',
        contentType: 'application/json',  
        success: function (response) {
            if (response['code'] == 1001) {
                alert("[Lỗi] Không nhận được phản hồi từ server, vui lòng kiểm tra lại!");
            }
            list_class = response['list_class'];
            console.log(list_class);
            document.getElementById("class_crop1").innerHTML ="Class: " + list_class[0];
            document.getElementById("class_crop2").innerHTML ="Class: " + list_class[1];
            document.getElementById("class_crop3").innerHTML ="Class: " + list_class[2];
            document.getElementById("class_crop4").innerHTML ="Class: " + list_class[3];
        }
    }).done(function() {
        
    }).fail(function() {
        alert('Fail!');
    });
}



// call function
// setInterval(getVehicleClass, 1000);