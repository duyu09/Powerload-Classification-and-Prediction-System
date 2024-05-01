const app = getApp()
Page({
  /**
   * 页面的初始数据
   */
  data: {
    loginBtnState: true,
    username: "",
    password: ""
  },
  usernameInput: function (e) {

    var val = e.detail.value;
    if (val != '') {
      this.setData({
        username: val
      })
      if (this.data.password != "") {
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
      if (this.data.username != "") {
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
  //点击登录 提交表单
  login: function (e) {
    var that = this
    const md5 = require('./md5.js')
    this.setData({
      password:md5.hexMD5(that.data.password)
    })
    wx.request({
      // /10.106.17.36:2000/api/login127.0.0.1:5000
      url: app.globalData.url + '/api/login',
      data: {
        phone_number: that.data.username,
        password: that.data.password
      },
      method: 'POST',
      header: {
      },
      success: function (res) {//res是接收后台返回给前台的数据   
        console.log(res.data)
        if (res.data.code == 1) {
          wx.showToast({
            icon: "error",
            title: '登录失败，请重试',
            duration: 2000
          })
        }
        else {
          console.log(res.data);
          app.globalData.token = res.data.token;
          app.globalData.username = that.data.username;
          app.globalData.status = "user";
          wx.showToast({
            title: '登录成功',
            duration: 2000,//2秒
            success: function () {
                wx.switchTab({
                  url: '../dist/dist',
                })
            }
          })
        }
      },
      fail: function (res) {
        console.log("发送失败");
        // app.globalData.status = "user"
        //   wx.switchTab({
        //     url: '../dist/dist',
        //   })
      }
    })
  },
  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {

    wx.setNavigationBarTitle({
      title: '用户登录',
    })
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})