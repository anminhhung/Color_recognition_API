function onLoad(){
    // drawImageOCR("/stream");
    // initAllCam('cam1');
    // initAllCam('cam2');


    // drawImageOCR("/stream");
    // drawImageOCR("/stream?camname=cam2");
}

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
        // document.getElementById("streamcam").src = "http://service.aiclub.cs.uit.edu.vn/vehicles_counting/stream1"
        document.getElementById("streamcam").src = "http://192.168.28.75:7778/stream1"
    } 
    else {
        // document.getElementById("streamcam").src = "http://service.aiclub.cs.uit.edu.vn/vehicles_counting/stream2"
        document.getElementById("streamcam").src = "http://192.168.28.75:7778/stream2"
    }
    
    console.log("camid: " + camname);
    initAllCam(camname);
}
)

if (!!window.EventSource) {
    var source = new EventSource('http://service.aiclub.cs.uit.edu.vn/vehicles_counting/class_vehicle');
    source.onmessage = function(e) {
      $("#data1").text(e.data);
    }
}