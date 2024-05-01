const app = getApp()
Page({

  /**
   * 页面的初始数据
   */
  data: {
    loginBtnState: true,
    username: "",
    password: "",
    password2: "",
    roomNum:0,
    roomNumList:["101","102","103"]
  },
  usernameInput: function (e) {

    var val = e.detail.value;
    if (val != '') {
      this.setData({
        username: val
      })
      if (this.data.password != "" & this.data.password2 != "") {
        this.setData({
          loginBtnState: false
        })
      }
    }
    else {
      this.setData({
        loginBtnState: true
      })
    }
  },
  passwordInput: function (e) {
    var val = e.detail.value;
    if (val != '') {
      this.setData({
        password: val
      })
      if (this.data.username != "" & this.data.password2 != "") {
        this.setData({
          loginBtnState: false
        })
      }
    }
    else {
      this.setData({
        loginBtnState: true
      })
    }
  },

  passwordInput2: function (e) {
    var val = e.detail.value;
    if (val != '') {
      this.setData({
        password2: val
      })
      if (this.data.password2 != "" & this.data.username != "") {
        this.setData({
          loginBtnState: false
        })
      }
    }
    else {
      this.setData({
        loginBtnState: true
      })
    }
  },
  bindPickerChange: function(e) {
    console.log('picker发送选择改变，携带值为', e.detail.value)
    this.setData({
      roomNum: e.detail.value
    })
  },
  //点击确认 提交表单
  register: function (e) {
    var that = this
    const md5 = require('./md5.js')
    this.setData({
      password:md5.hexMD5(that.data.password)
    })
    wx.request({
      // /10.106.17.36:2000/api/login
      url: app.globalData.url + '/api/register',
      data: {
        phone_number: that.data.username,
        password: that.data.password,
        room:that.data.roomNumList[that.data.roomNum]
      },
      method: 'POST',
      header: {
      },
      success: function (res) {//res是接收后台返回给前台的数据    
        console.log(res.data);
        //  console.log("地址是%s%s",app.globalData.village,app.globalData.housenumber)
        if (res.data.code == 0) {
          wx.showToast({
            title: '注册成功',
            duration: 2000,//2秒
            success: function () {
              setTimeout(function(){
                wx.switchTab({
                  url: '../login/login',
                })
              },2000)

            }
          })
        }
        else if (res.data.code == 1) {
          wx.showToast({
            title: '账号已存在',
            duration: 2000,
            icon:"error"
          })
        }
        else {
          wx.showToast({
            title: '请重新注册',
            duration: 2000
          })
        }
      },
      fail: function (res) {
        console.log("发送失败");
      }
    })
  },
  /**
   * 生命周期函数--监听页面加载
   */
  onLoad(options) {
    wx.setNavigationBarTitle({
      title: '用户注册',
    })
  },



  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady() {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow() {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide() {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload() {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh() {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom() {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage() {

  }
})