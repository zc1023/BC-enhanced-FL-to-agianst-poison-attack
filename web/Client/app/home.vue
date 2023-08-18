<template>
  <div>
    <input v-model="parameters.learning_rate" placeholder="Learning rate">
    <!-- 在这里添加更多的输入框来获取更多的参数 -->
    <button @click="startTraining">Start training</button>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      parameters: {
        learning_rate: 0.01,
        // 在这里添加更多的参数
      },
    };
  },
  methods: {
    async startTraining() {
      await axios.post('http://localhost:5000/train', this.parameters);
    },
  },
};
</script>
