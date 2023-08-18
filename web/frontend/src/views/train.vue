<template>
  <div>
    <h2>联邦学习参数配置</h2>
    <form @submit.prevent="submitForm">
      <label for="datasets">Datasets:</label>
      <input type="text" id="datasets" v-model="form.datasets">
      
      <label for="optimizer">Optimizer:</label>
      <input type="text" id="optimizer" v-model="form.optimizer">

      <label for="seed">Seed:</label>
      <input type="number" id="seed" v-model="form.seed">

      <label for="model">Model:</label>
      <input type="text" id="model" v-model="form.model">

      <!-- 你可以根据需要添加更多的输入框 -->

      <button type="submit">开始训练</button>
    </form>
  </div>
</template>

<script>
export default {
  data() {
    return {
      form: {
        datasets: '',
        optimizer: '',
        seed: 0,
        model: '',
        // ... 更多的属性
      }
    }
  },
  methods: {
    submitForm() {
      // 在这里，你可以处理表单提交事件
      // 例如，你可以将表单数据发送到后端
      console.log(this.form);
    }
  }
}
</script>
